package tokenizer

import (
	"slices"
	"sync"
	"testing"
)

// The tiny tokenizer merges, in rank order: "b"+"c" (id 15), "a"+"b" (id 16)
// and "a"+"bc" (id 17). The expected ids in this file come from the Python
// tokenizers library loading testdata/tiny/tokenizer.json.

func TestEncodeTinyMergesByRank(t *testing.T) {
	tok := loadTiny(t)
	tests := []struct {
		name string
		text string
		want []int32
	}{
		// Left to right, "ab" would form first and leave "c" behind.
		{"lower rank on the right goes first", "abc", []int32{17}},
		{"single merge", "ab", []int32{16}},
		{"merge is not available twice", "aab", []int32{3, 16}},
		{"overlapping candidates", "bbc", []int32{4, 15}},
		{"merged pair meets its left neighbour", "ababc", []int32{16, 17}},
		{"merged pair meets nothing on its right", "abcbc", []int32{17, 15}},
		{"repeated word", "abcabc", []int32{17, 17}},
		{"pieces merge separately", "ab ab", []int32{16, 14, 16}},
		{"space leads its piece", " ab", []int32{14, 16}},
		{"added token separates merges", "ab<|im_start|>bc", []int32{16, 18, 15}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tok.Encode(tt.text); !slices.Equal(got, tt.want) {
				t.Errorf("Encode(%q) = %v, want %v", tt.text, got, tt.want)
			}
		})
	}
}

// tokenizeRow is one row of tokenize.jsonl: the reference's ids for Text.
type tokenizeRow struct {
	Text string  `json:"text"`
	IDs  []int32 `json:"ids"`
}

func TestEncodeMatchesGolden(t *testing.T) {
	for _, family := range families {
		t.Run(family, func(t *testing.T) {
			tok := loadFamily(t, family)
			rows := readGolden[tokenizeRow](t, family, "tokenize.jsonl")

			mismatches := 0
			for i, row := range rows {
				got := tok.Encode(row.Text)
				if slices.Equal(got, row.IDs) {
					continue
				}
				if mismatches == 0 {
					t.Errorf("first mismatch at row %d, text %+q:\n got %v\nwant %v", i, row.Text, got, row.IDs)
				}
				mismatches++
			}
			if mismatches > 0 {
				t.Errorf("%d of %d rows differ", mismatches, len(rows))
			}
		})
	}
}

// Without an unknown token the reference drops what the vocabulary cannot
// spell, and merges across the gap. The tiny vocabulary has no "x", "y" or "z".
func TestEncodeDropsBytesOutsideVocabulary(t *testing.T) {
	tok := loadTiny(t)
	for text, want := range map[string][]int32{"xyz": {}, "axb": {16}} {
		if got := tok.Encode(text); !slices.Equal(got, want) {
			t.Errorf("Encode(%q) = %v, want %v", text, got, want)
		}
	}
}

// Run with -race: the tokenizer's tables are shared by all goroutines.
func TestEncodeIsSafeForConcurrentUse(t *testing.T) {
	for _, family := range families {
		t.Run(family, func(t *testing.T) {
			tok := loadFamily(t, family)
			rows := readGolden[tokenizeRow](t, family, "tokenize.jsonl")

			const goroutines = 16
			mismatches := make([]int, goroutines)
			var wg sync.WaitGroup
			for g := range goroutines {
				wg.Go(func() {
					for _, row := range rows {
						if !slices.Equal(tok.Encode(row.Text), row.IDs) {
							mismatches[g]++
						}
					}
				})
			}
			wg.Wait()

			for g, n := range mismatches {
				if n > 0 {
					t.Errorf("goroutine %d: %d of %d rows differ", g, n, len(rows))
				}
			}
		})
	}
}

// A Go string can hold bytes that are not UTF-8, which the reference's input
// never does. Each such byte is encoded as U+FFFD.
func TestEncodeReadsInvalidUTF8AsReplacementCharacter(t *testing.T) {
	for _, family := range families {
		t.Run(family, func(t *testing.T) {
			tok := loadFamily(t, family)

			got, want := tok.Encode("a\xffb\xe4\xbd"), tok.Encode("a\ufffdb\ufffd\ufffd")
			if !slices.Equal(got, want) {
				t.Errorf("Encode of invalid bytes = %v, want %v as for U+FFFD in their place", got, want)
			}
		})
	}
}
