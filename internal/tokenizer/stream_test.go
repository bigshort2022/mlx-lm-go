package tokenizer

import (
	"slices"
	"strings"
	"testing"
	"unicode/utf8"
)

// In the tiny tokenizer ids 13 and 12 are the bytes C3 and A9, which together
// are "é"; id 3 is "a".
func TestStreamTiny(t *testing.T) {
	tok := loadTiny(t)
	tests := []struct {
		name  string
		ids   []int32
		adds  []string // what each Add returns
		flush string
	}{
		{"whole characters pass straight through", []int32{3, 14, 4}, []string{"a", " ", "b"}, ""},
		{"special tokens are text", []int32{18, 3}, []string{"<|im_start|>", "a"}, ""},
		{"split character waits for its last byte", []int32{13, 12}, []string{"", "\u00e9"}, ""},
		{"held byte turns out to be invalid", []int32{13, 3}, []string{"", "\ufffda"}, ""},
		{"held byte gives way to the next lead byte", []int32{13, 13, 12}, []string{"", "\ufffd", "\u00e9"}, ""},
		{"flush gives up on an unfinished character", []int32{3, 13}, []string{"a", ""}, "\ufffd"},
		{"unknown ids add nothing", []int32{99, -1}, []string{"", ""}, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stream := tok.NewStream()
			var adds []string
			for _, id := range tt.ids {
				adds = append(adds, stream.Add(id))
			}
			if !slices.Equal(adds, tt.adds) {
				t.Errorf("Add results = %+q, want %+q", adds, tt.adds)
			}
			if got := stream.Flush(); got != tt.flush {
				t.Errorf("Flush() = %+q, want %+q", got, tt.flush)
			}
			if got := stream.Flush(); got != "" {
				t.Errorf("second Flush() = %+q, want nothing left", got)
			}
		})
	}
}

func TestStreamConcatenationEqualsDecode(t *testing.T) {
	for _, family := range families {
		t.Run(family, func(t *testing.T) {
			tok := loadFamily(t, family)
			rows := readGolden[decodeRow](t, family, "decode.jsonl")

			mismatches := 0
			for i, row := range rows {
				stream := tok.NewStream()
				var streamed strings.Builder
				for _, id := range row.IDs {
					streamed.WriteString(stream.Add(id))
				}
				streamed.WriteString(stream.Flush())

				want := tok.Decode(row.IDs)
				if streamed.String() == want {
					continue
				}
				if mismatches == 0 {
					t.Errorf("first mismatch at row %d, ids %v:\nstreamed %+q\n  Decode %+q", i, row.IDs, streamed.String(), want)
				}
				mismatches++
			}
			if mismatches > 0 {
				t.Errorf("%d of %d rows differ", mismatches, len(rows))
			}
		})
	}
}

func TestStreamNeverEmitsPartialCharacters(t *testing.T) {
	for _, family := range families {
		t.Run(family, func(t *testing.T) {
			tok := loadFamily(t, family)
			rows := readGolden[decodeRow](t, family, "decode.jsonl")

			checked := 0
			for i, row := range rows {
				if strings.Contains(row.Text, "\ufffd") {
					continue // the ids themselves stop part-way through a character
				}
				checked++
				stream := tok.NewStream()
				for j, id := range row.IDs {
					got := stream.Add(id)
					if !utf8.ValidString(got) || strings.Contains(got, "\ufffd") {
						t.Fatalf("row %d: Add of id %d (number %d of %v) = %+q, want whole characters only", i, id, j, row.IDs, got)
					}
				}
			}
			if checked == 0 {
				t.Fatal("every row has U+FFFD; nothing was checked")
			}
		})
	}
}

func TestStreamHoldsBackSplitEmoji(t *testing.T) {
	// Both Qwen vocabularies have a single token for U+1F600 (grinning face),
	// so U+1FAE0 (melting face), which takes three, is checked as well.
	emoji := []string{"\U0001F600", "\U0001FAE0"}

	for _, family := range families {
		t.Run(family, func(t *testing.T) {
			tok := loadFamily(t, family)

			split := false
			for _, text := range emoji {
				ids := tok.Encode(text)
				split = split || len(ids) > 1

				stream := tok.NewStream()
				for i, id := range ids {
					want := ""
					if i == len(ids)-1 {
						want = text
					}
					if got := stream.Add(id); got != want {
						t.Errorf("%+q is ids %v: Add number %d = %+q, want %+q", text, ids, i, got, want)
					}
				}
			}
			if !split {
				t.Error("no emoji took more than one token; nothing was held back")
			}
		})
	}
}
