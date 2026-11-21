package tokenizer

import "strings"

// Stream decodes the ids of one generation as they arrive, so that the text
// it has returned so far plus Flush always equals Decode of the ids so far.
// It must not be used by more than one goroutine.
type Stream struct {
	tokenizer *Tokenizer
	// pending holds the bytes of a character whose remaining bytes have not
	// arrived yet. It is empty between characters.
	pending []byte
}

// NewStream returns a streaming decoder for one generation.
func (t *Tokenizer) NewStream() *Stream {
	return &Stream{tokenizer: t}
}

// Add consumes one id and returns the text that is now complete. Bytes at the
// end that begin a character without finishing it are held back until the
// ids that finish it arrive, so an id that leaves a character unfinished and
// completes nothing before it returns "".
func (s *Stream) Add(id int32) string {
	s.pending = append(s.pending, s.tokenizer.bytesOf(id)...)
	var text strings.Builder
	consumed := appendText(&text, s.pending)
	s.pending = s.pending[:copy(s.pending, s.pending[consumed:])]
	return text.String()
}

// Flush returns whatever is still held back, with invalid bytes as U+FFFD.
func (s *Stream) Flush() string {
	text := bytesToText(s.pending)
	s.pending = s.pending[:0]
	return text
}
