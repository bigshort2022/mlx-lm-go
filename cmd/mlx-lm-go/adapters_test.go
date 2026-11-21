package main

import (
	"slices"
	"testing"

	"github.com/bigshort2022/mlx-lm-go/internal/hub"
	"github.com/bigshort2022/mlx-lm-go/internal/tokenizer"
)

// referenceTokenizer loads the tokenizer of the reference model.
func referenceTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	needModel(t)
	dir, err := hub.Resolve(referenceModel)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := tokenizer.Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

// The expected pieces are what mlx_lm's BPEStreamingDetokenizer returns for
// the same tokens, one piece per token and then the rest.
func TestDetokenizerReturnsThePiecesOfTheReference(t *testing.T) {
	tok := referenceTokenizer(t)
	tests := []struct {
		name   string
		text   string
		pieces []string
		rest   string
	}{
		{"a space waits for the next token", "1, 2", []string{"1", ",", "", " 2"}, ""},
		{"a space at the start is dropped", " 4, 5", []string{"", "4", ",", "", " 5"}, ""},
		{"a space at the end comes with the rest", "a ", []string{"a", ""}, " "},
		{"a space alone is dropped", " ", []string{""}, ""},
		{"only one space at the start is dropped", "  a", []string{"", " a"}, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d := newDetokenizer(tok)
			var pieces []string
			for _, token := range tok.Encode(tt.text) {
				pieces = append(pieces, d.Add(token))
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
