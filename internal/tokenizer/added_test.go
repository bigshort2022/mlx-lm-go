package tokenizer

import (
	"slices"
	"testing"
)

// The expected ids in this file come from the Python tokenizers library
// loading testdata/tiny/tokenizer.json.

func TestEncodeSplitsAddedTokens(t *testing.T) {
	tok := loadTiny(t)
	tests := []struct {
		name string
		text string
		want []int32
	}{
		{"alone", "<|im_end|>", []int32{19}},
		{"between ordinary text", "a<|im_start|>c", []int32{3, 18, 5}},
		{"adjacent", "<|im_start|><|im_end|>", []int32{18, 19}},
		{"partial token is ordinary text", "<|im_start", []int32{0, 11, 6, 7, 2, 9, 10, 3, 8, 10}},
		{"after a partial token", "<|im_<|im_end|>", []int32{0, 11, 6, 7, 2, 19}},
		{"after its own first character", "<<|im_end|>", []int32{0, 19}},
		{"longest of two starting together", "c<a>bc<a>", []int32{5, 22, 5, 21}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tok.Encode(tt.text); !slices.Equal(got, tt.want) {
				t.Errorf("Encode(%q) = %v, want %v", tt.text, got, tt.want)
			}
		})
	}
}

// The tiny tokenizer has an added token spelled with a combining accent,
// "<e\u0301>". NFC composes "e\u0301" into "é" (bytes C3 A9, ids 13 and 12),
// so these cases tell matching before normalisation from matching after it.
func TestAddedTokensAreMatchedBeforeNormalization(t *testing.T) {
	tok := loadTiny(t)
	tests := []struct {
		name string
		text string
		want []int32
	}{
		{"token keeps its spelling, text around it is composed", "e\u0301<e\u0301>", []int32{13, 12, 20}},
		{"composed spelling is not the token", "<\u00e9>", []int32{0, 13, 12, 1}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tok.Encode(tt.text); !slices.Equal(got, tt.want) {
				t.Errorf("Encode(%+q) = %v, want %v", tt.text, got, tt.want)
			}
		})
	}
}
