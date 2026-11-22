package tokenizer

import (
	"slices"
	"testing"
)

// pretokenizeRow is one row of pretokenize.jsonl: the reference normalizer's
// output for Text and the reference pre-tokenizer's split of that output.
type pretokenizeRow struct {
	Text       string   `json:"text"`
	Normalized string   `json:"normalized"`
	Pieces     []string `json:"pieces"`
}

func splitToStrings(tok *Tokenizer, normalized string) []string {
	var pieces []string
	for piece := range tok.splitter.pieces([]rune(normalized)) {
		pieces = append(pieces, string(piece))
	}
	return pieces
}

func TestPreTokenizeMatchesGolden(t *testing.T) {
	for _, family := range families {
		t.Run(family, func(t *testing.T) {
			tok := loadFamily(t, family)
			rows := readGolden[pretokenizeRow](t, family, "pretokenize.jsonl")

			mismatches := 0
			for i, row := range rows {
				normalized := normalize(row.Text)
				pieces := splitToStrings(tok, normalized)
				if normalized == row.Normalized && slices.Equal(pieces, row.Pieces) {
					continue
				}
				if mismatches == 0 {
					t.Errorf("first mismatch at row %d, text %+q:\n normalized %+q\n      want %+q\n     pieces %+q\n       want %+q",
						i, row.Text, normalized, row.Normalized, pieces, row.Pieces)
				}
				mismatches++
			}
			if mismatches > 0 {
				t.Errorf("%d of %d rows differ", mismatches, len(rows))
			}
		})
	}
}

// The tiny tokenizer carries Qwen2.5's split pattern, so these rows, copied
// from testdata/qwen2/pretokenize.jsonl, check the split without the cache.
func TestSplitTiny(t *testing.T) {
	tok := loadTiny(t)
	tests := []struct {
		text string
		want []string
	}{
		{"", nil},
		{"Hello, world!", []string{"Hello", ",", " world", "!"}},
		{"12345", []string{"1", "2", "3", "4", "5"}},
		{"trailing spaces   ", []string{"trailing", " spaces", "   "}},
		{"two  spaces  between  words", []string{"two", " ", " spaces", " ", " between", " ", " words"}},
		{"mixed \n \t whitespace \n\n  runs", []string{"mixed", " \n", " \t", " whitespace", " \n\n", " ", " runs"}},
		{"don't I'll they're we've I'm he'd", []string{"don", "'t", " I", "'ll", " they", "'re", " we", "'ve", " I", "'m", " he", "'d"}},
	}
	for _, tt := range tests {
		if got := splitToStrings(tok, tt.text); !slices.Equal(got, tt.want) {
			t.Errorf("pieces of %q = %q, want %q", tt.text, got, tt.want)
		}
	}
}

func TestNormalizeComposesCharacters(t *testing.T) {
	if got, want := normalize("e\u0301 a\u030a\u0301"), "\u00e9 \u01fb"; got != want {
		t.Errorf("normalize = %+q, want %+q", got, want)
	}
}

// The Qwen patterns match every character and never match nothing. Other
// patterns must still split like the reference: text between matches is a
// piece, and an empty match cuts the text without becoming a piece. The
// expected pieces come from the Python tokenizers library.
func TestSplitIsolatesMatches(t *testing.T) {
	tests := []struct {
		pattern string
		text    string
		want    []string
	}{
		{"b+", "abbcba", []string{"a", "bb", "c", "b", "a"}},
		{"b+", "bab", []string{"b", "a", "b"}},
		{"b+", "xyz", []string{"xyz"}},
		{"x*", "abc", []string{"a", "b", "c"}},
		{"x*", "xxaxx", []string{"xx", "a", "xx"}},
		{"(?=b)", "abab", []string{"a", "ba", "b"}},
	}
	for _, tt := range tests {
		splitter, err := newSplitter(tt.pattern)
		if err != nil {
			t.Fatal(err)
		}
		tok := &Tokenizer{splitter: splitter}
		if got := splitToStrings(tok, tt.text); !slices.Equal(got, tt.want) {
			t.Errorf("pieces of %q split by %q = %q, want %q", tt.text, tt.pattern, got, tt.want)
		}
	}
}
