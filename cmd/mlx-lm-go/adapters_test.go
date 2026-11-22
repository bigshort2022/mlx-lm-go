package main

import (
	"path/filepath"
	"slices"
	"testing"

	"github.com/bigshort2022/mlx-lm-go/internal/tokenizer"
)

// TestDetokenizerReturnsThePiecesOfTheReference pins the detokenizer to
// BPEStreamingDetokenizer of mlx-lm 0.32.2: every expectation is what that
// class returned for the same tokens, one piece per token and then the rest
// at finalize. The vocabulary in testdata/detokenizer has one token per
// digit, the comma, the space (Ġ), "a", and the two bytes of "é" (Ã and ©)
// as a token each.
func TestDetokenizerReturnsThePiecesOfTheReference(t *testing.T) {
	tok, err := tokenizer.Load(filepath.Join("testdata", "detokenizer"))
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name   string
		tokens []string
		pieces []string
		rest   string
	}{
		// A plain decode has produced "1, 2, " when the 3 arrives. Here the
		// 3 arrives as " 3", so a stop at "3" leaves "1, 2,", which is the
		// text of testdata/server/chat_stop_string.json.
		{"a space waits for the token after it",
			[]string{"1", ",", "Ġ", "2", ",", "Ġ", "3"},
			[]string{"1", ",", "", " 2", ",", "", " 3"}, ""},
		// A plain decode yields " 4, 5, 6"; testdata/server/completions.json
		// has "4, 5, 6".
		{"one space at the start of the text is dropped",
			[]string{"Ġ", "4", ",", "Ġ", "5", ",", "Ġ", "6"},
			[]string{"", "4", ",", "", " 5", ",", "", " 6"}, ""},
		{"of two spaces at the start one is dropped",
			[]string{"Ġ", "Ġ", "a"},
			[]string{"", "", " a"}, ""},
		{"a space alone is dropped",
			[]string{"Ġ"},
			[]string{""}, ""},
		{"a space at the end comes with the rest",
			[]string{"a", "Ġ"},
			[]string{"a", ""}, " "},
		{"a character split across tokens comes when it is complete",
			[]string{"a", "Ã", "©"},
			[]string{"a", "", "é"}, ""},
		{"a space before a split character waits for the whole character",
			[]string{"a", "Ġ", "Ã", "©"},
			[]string{"a", "", "", " é"}, ""},
		{"a character that is never completed is replaced at the end",
			[]string{"a", "Ã"},
			[]string{"a", ""}, "\ufffd"},
		{"a character that the next token does not complete is replaced",
			[]string{"Ã", "a"},
			[]string{"", "\ufffda"}, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d := newBPEStreamingDetokenizer(tok)
			var pieces []string
			for _, token := range tt.tokens {
				id, ok := tok.TokenID(token)
				if !ok {
					t.Fatalf("the test vocabulary has no token %q", token)
				}
				pieces = append(pieces, d.Add(id))
			}
			if !slices.Equal(pieces, tt.pieces) {
				t.Errorf("pieces = %q, want %q", pieces, tt.pieces)
			}
			if rest := d.Flush(); rest != tt.rest {
				t.Errorf("rest = %q, want %q", rest, tt.rest)
			}
		})
	}
}
