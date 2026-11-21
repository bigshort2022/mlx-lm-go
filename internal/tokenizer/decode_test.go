package tokenizer

import "testing"

// The expected text in TestDecodeTiny comes from the Python tokenizers
// library loading testdata/tiny/tokenizer.json. Ids 13 and 12 are the two
// bytes of "é", C3 and A9.
func TestDecodeTiny(t *testing.T) {
	tok := loadTiny(t)
	tests := []struct {
		name string
		ids  []int32
		want string
	}{
		{"nothing", nil, ""},
		{"byte-level space", []int32{3, 14, 4}, "a b"},
		{"character split over two tokens", []int32{13, 12}, "\u00e9"},
		{"lead byte alone", []int32{13}, "\ufffd"},
		{"continuation byte alone", []int32{12}, "\ufffd"},
		{"lead byte before ASCII", []int32{13, 3}, "\ufffda"},
		{"lead byte before a whole character", []int32{13, 13, 12}, "\ufffd\u00e9"},
		{"lead byte at the end", []int32{3, 13}, "a\ufffd"},
		{"special tokens are kept", []int32{18, 3, 19}, "<|im_start|>a<|im_end|>"},
		{"added token outside the byte alphabet", []int32{20}, "<e\u0301>"},
		{"unknown ids are skipped", []int32{3, 99, -1, 4}, "ab"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tok.Decode(tt.ids); got != tt.want {
				t.Errorf("Decode(%v) = %+q, want %+q", tt.ids, got, tt.want)
			}
		})
	}
}

// The expected text comes from the reference: each byte sequence was decoded
// by the Python tokenizers library through Qwen2.5's 256 single-byte tokens.
// It replaces each maximal prefix of a valid sequence with one U+FFFD, and
// every other invalid byte with its own.
func TestBytesToTextReplacesInvalidUTF8LikeReference(t *testing.T) {
	tests := []struct {
		name string
		data string
		want string
	}{
		{"valid three-byte character", "\xe4\xbd\xa0", "\u4f60"},
		{"valid four-byte character", "\xf0\x9f\x98\x80", "\U0001f600"},
		{"lowest four-byte character", "\xf0\x90\x80\x80", "\U00010000"},
		{"highest character", "\xf4\x8f\xbf\xbf", "\U0010ffff"},
		{"last character before the surrogates", "\xed\x9f\xbf", "\ud7ff"},
		{"encoded U+FFFD", "\xef\xbf\xbd", "\ufffd"},

		{"two-byte lead at the end", "\xc2", "\ufffd"},
		{"three-byte lead at the end", "\xef", "\ufffd"},
		{"four-byte lead at the end", "\xf0", "\ufffd"},
		{"two of three bytes at the end", "\xe4\xbd", "\ufffd"},
		{"two of four bytes at the end", "\xf0\x9f", "\ufffd"},
		{"three of four bytes at the end", "\xf0\x9f\x98", "\ufffd"},
		{"text then a lead byte", "ab\xc3", "ab\ufffd"},
		{"character then a lead byte", "\xc3\xa9\xc3", "\u00e9\ufffd"},

		{"two-byte lead before ASCII", "\xc2A", "\ufffdA"},
		{"two of three bytes before ASCII", "\xe4\xbdA", "\ufffdA"},
		{"three of four bytes before ASCII", "\xf0\x9f\x98A", "\ufffdA"},
		{"three of four bytes before a lead byte", "\xf1\x80\x80\xe4", "\ufffd\ufffd"},
		{"lead byte before a whole character", "\xe4\xe4\xbd\xa0", "\ufffd\u4f60"},
		{"two of three bytes between text", "a\xe4\xbd\xe4\xbd\xa0b", "a\ufffd\u4f60b"},
		{"three of four bytes before a whole character", "\xf0\x9f\x98\xf0\x9f\x98\x80", "\ufffd\U0001f600"},

		{"continuation byte alone", "\x80", "\ufffd"},
		{"two continuation bytes", "\x80\x80", "\ufffd\ufffd"},
		{"overlong two-byte lead", "\xc0\xaf", "\ufffd\ufffd"},
		{"overlong three-byte sequence", "\xe0\x80\xaf", "\ufffd\ufffd\ufffd"},
		{"overlong four-byte sequence", "\xf0\x80\x80\x80", "\ufffd\ufffd\ufffd\ufffd"},
		{"surrogate", "\xed\xa0\x80", "\ufffd\ufffd\ufffd"},
		{"beyond the highest character", "\xf4\x90\x80\x80", "\ufffd\ufffd\ufffd\ufffd"},
		{"byte that is never a lead", "\xf5", "\ufffd"},
		{"byte that is never valid", "\xff", "\ufffd"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := bytesToText([]byte(tt.data)); got != tt.want {
				t.Errorf("bytesToText(%+q) = %+q, want %+q", tt.data, got, tt.want)
			}
		})
	}
}

// decodeRow is one row of decode.jsonl: the reference's text for IDs. Some
// rows end part-way through a character, which the reference decodes to U+FFFD.
type decodeRow struct {
	IDs  []int32 `json:"ids"`
	Text string  `json:"text"`
}

func TestDecodeMatchesGolden(t *testing.T) {
	for _, family := range families {
		t.Run(family, func(t *testing.T) {
			tok := loadFamily(t, family)
			rows := readGolden[decodeRow](t, family, "decode.jsonl")

			mismatches := 0
			for i, row := range rows {
				got := tok.Decode(row.IDs)
				if got == row.Text {
					continue
				}
				if mismatches == 0 {
					t.Errorf("first mismatch at row %d, ids %v:\n got %+q\nwant %+q", i, row.IDs, got, row.Text)
				}
				mismatches++
			}
			if mismatches > 0 {
				t.Errorf("%d of %d rows differ", mismatches, len(rows))
			}
		})
	}
}
