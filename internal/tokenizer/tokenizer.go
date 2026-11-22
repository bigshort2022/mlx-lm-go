// Package tokenizer reads a Hugging Face tokenizer.json and encodes, decodes
// and stream-decodes text the way the Python tokenizers library does for the
// byte-level BPE tokenizers of the Qwen family.
//
// Encoding cuts added tokens out of the text, NFC-normalises what is left,
// splits it with the file's own regex, and runs BPE on the bytes of each
// piece. Decoding maps ids back to bytes and replaces invalid UTF-8 the way
// the reference does.
package tokenizer

// Tokenizer converts between text and token ids. It is immutable after Load
// and safe for concurrent use.
type Tokenizer struct {
	// ids maps every token, base vocabulary and added, to its id.
	ids map[string]int32
	// eos and eosID are the end-of-sequence token and its id. They are set
	// only when hasEOS is true.
	eos    string
	eosID  int32
	hasEOS bool
	// added finds added tokens in raw text, before anything else looks at it.
	added *addedTokens
	// splitter cuts normalised text into the pieces BPE runs on.
	splitter splitter
	// bpe turns each piece into ids.
	bpe *bpe
	// decodeTable holds, indexed by id, the bytes each token decodes to.
	decodeTable []string
}

// VocabSize returns the number of tokens: the base vocabulary plus added tokens.
func (t *Tokenizer) VocabSize() int {
	return len(t.ids)
}

// TokenID returns the id of a token given exactly as it is spelled in
// tokenizer.json, such as "<|im_end|>" or "Ġthe".
func (t *Tokenizer) TokenID(token string) (int32, bool) {
	id, ok := t.ids[token]
	return id, ok
}

// EOSToken returns the end-of-sequence token named by tokenizer_config.json
// "eos_token" and its id. ok is false when the model directory names none.
func (t *Tokenizer) EOSToken() (token string, id int32, ok bool) {
	return t.eos, t.eosID, t.hasEOS
}

// Encode converts text to token ids. Added and special tokens that appear
// literally in the text, such as "<|im_start|>", become their single id.
// Bytes that are not valid UTF-8 are encoded as U+FFFD.
func (t *Tokenizer) Encode(text string) []int32 {
	var ids []int32
	merger := merger{model: t.bpe}
	for {
		start, end, id, found := t.added.find(text)
		if !found {
			return t.appendOrdinary(ids, text, &merger)
		}
		ids = t.appendOrdinary(ids, text[:start], &merger)
		ids = append(ids, id)
		text = text[end:]
	}
}

// appendOrdinary appends the ids of text that holds no added token:
// normalise, split into pieces, then run BPE on each piece.
func (t *Tokenizer) appendOrdinary(ids []int32, text string, merger *merger) []int32 {
	for piece := range t.splitter.pieces([]rune(normalize(text))) {
		ids = merger.appendPiece(ids, piece)
	}
	return ids
}
