package tokenizer

import (
	"fmt"
	"strings"
)

// runeBytes inverts byteRunes: the byte each character of the byte-level
// alphabet stands for.
var runeBytes = func() map[rune]byte {
	inverse := make(map[rune]byte, len(byteRunes))
	for b, r := range byteRunes {
		inverse[r] = byte(b)
	}
	return inverse
}()

// tokenBytes returns the bytes a token decodes to. Each character of a
// byte-level token stands for one byte. A token with a character outside the
// byte-level alphabet decodes to its own UTF-8 bytes instead, as in the
// reference's ByteLevel decoder.
func tokenBytes(token string) string {
	decoded := make([]byte, 0, len(token))
	for _, r := range token {
		b, ok := runeBytes[r]
		if !ok {
			return token
		}
		decoded = append(decoded, b)
	}
	return string(decoded)
}

// newDecodeTable returns, indexed by id, the bytes every token decodes to. An
// id that no token has decodes to nothing. An added token takes precedence
// over a base vocabulary token with the same id.
//
// Ids must be below the number of tokens, which holds for any vocabulary
// numbered without gaps and keeps a corrupt id from sizing the table.
func newDecodeTable(vocab map[string]int32, added []addedTokenFile) ([]string, error) {
	table := make([]string, len(vocab)+len(added))
	set := func(token string, id int32) error {
		if id < 0 || int(id) >= len(table) {
			return fmt.Errorf("token %q has id %d, outside 0..%d", token, id, len(table)-1)
		}
		table[id] = tokenBytes(token)
		return nil
	}
	for token, id := range vocab {
		if err := set(token, id); err != nil {
			return nil, err
		}
	}
	for _, token := range added {
		if err := set(token.Content, token.ID); err != nil {
			return nil, err
		}
	}
	return table, nil
}

// Decode converts ids to text, including special tokens. Invalid UTF-8 becomes U+FFFD.
func (t *Tokenizer) Decode(ids []int32) string {
	var data []byte
	for _, id := range ids {
		data = append(data, t.bytesOf(id)...)
	}
	return bytesToText(data)
}

// bytesOf returns the bytes id decodes to. Like the reference, it has nothing
// for an id that is not a token.
func (t *Tokenizer) bytesOf(id int32) string {
	if id < 0 || int(id) >= len(t.decodeTable) {
		return ""
	}
	return t.decodeTable[id]
}

// replacement is U+FFFD, which stands in for bytes that are not UTF-8.
const replacement = "\ufffd"

// bytesToText converts decoded token bytes to text, replacing invalid UTF-8
// with U+FFFD. Bytes at the end that only begin a character are invalid too.
func bytesToText(data []byte) string {
	var text strings.Builder
	text.Grow(len(data))
	if consumed := appendText(&text, data); consumed < len(data) {
		text.WriteString(replacement)
	}
	return text.String()
}

// appendText appends data to text, with one U+FFFD in place of each invalid
// sequence. It stops at a trailing incomplete sequence, which later bytes
// might complete, and returns how many bytes of data it consumed.
func appendText(text *strings.Builder, data []byte) (consumed int) {
	written := 0 // data[written:i] is valid and waits to be appended
	for i := 0; i < len(data); {
		size, state := leadingSequence(data[i:])
		switch state {
		case incomplete:
			text.Write(data[written:i])
			return i
		case invalid:
			text.Write(data[written:i])
			text.WriteString(replacement)
			written = i + size
		}
		i += size
	}
	text.Write(data[written:])
	return len(data)
}

// sequenceState says what the bytes at the start of a buffer are.
type sequenceState int

const (
	complete   sequenceState = iota // one whole character
	invalid                         // bytes that no character contains in this order
	incomplete                      // the start of a character, cut off by the end of the buffer
)

// leadingSequence classifies the first bytes of data, which must not be
// empty, and returns how many bytes it classified.
//
// An invalid sequence is as long as the reference (Rust's from_utf8_lossy)
// makes it, since each one becomes a single U+FFFD: the longest run that
// starts a valid character if the first byte can start one, else one byte.
// Go's own decoding would report every byte of such a run separately.
func leadingSequence(data []byte) (size int, state sequenceState) {
	// A second byte must lie in low..high. That is narrower than the usual
	// 80..BF after the lead bytes that could otherwise begin an overlong
	// encoding, a surrogate, or a value beyond U+10FFFF.
	low, high := byte(0x80), byte(0xBF)
	switch lead := data[0]; {
	case lead < 0x80:
		return 1, complete
	case 0xC2 <= lead && lead <= 0xDF:
		size = 2
	case lead == 0xE0:
		size, low = 3, 0xA0
	case lead == 0xED:
		size, high = 3, 0x9F
	case 0xE1 <= lead && lead <= 0xEF:
		size = 3
	case lead == 0xF0:
		size, low = 4, 0x90
	case lead == 0xF4:
		size, high = 4, 0x8F
	case 0xF1 <= lead && lead <= 0xF3:
		size = 4
	default:
		return 1, invalid
	}
	for i := 1; i < size; i++ {
		if i == len(data) {
			return i, incomplete
		}
		if data[i] < low || high < data[i] {
			return i, invalid
		}
		low, high = 0x80, 0xBF
	}
	return size, complete
}
