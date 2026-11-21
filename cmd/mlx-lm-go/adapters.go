package main

import (
	"encoding/json"
	"strings"

	"github.com/bigshort2022/mlx-lm-go/internal/chattemplate"
	"github.com/bigshort2022/mlx-lm-go/internal/server"
	"github.com/bigshort2022/mlx-lm-go/internal/tokenizer"
)

// serverTokenizer makes a tokenizer the server.Tokenizer.
type serverTokenizer struct{ *tokenizer.Tokenizer }

var _ server.Tokenizer = serverTokenizer{}

func (t serverTokenizer) NewDetokenizer() server.Detokenizer {
	return newBPEStreamingDetokenizer(t.Tokenizer)
}

// thinkingTokens recognizes exact vocabulary entries. Qwen3 marks them as
// added tokens but not special tokens; Qwen2 only has their component pieces.
func (t serverTokenizer) thinkingTokens() *server.ThinkingTokens {
	start, hasStart := t.TokenID("<think>")
	end, hasEnd := t.TokenID("</think>")
	if !hasStart || !hasEnd {
		return nil
	}
	return &server.ThinkingTokens{Start: start, End: end}
}

// bpeStreamingDetokenizer turns the tokens of one generation into text. It is
// a port of BPEStreamingDetokenizer in mlx_lm/tokenizer_utils.py of mlx-lm
// 0.32.2 and returns the same pieces, so that streams, stop strings and
// completions come out as they do from mlx_lm.server. It differs from
// decoding one token after another in three ways:
//
//   - The token that is a single space is held back until the token after
//     it, so that " 3" arrives as one piece. The server drops whole pieces
//     when it meets a stop string, which is why "1, 2, 3" stopped at "3"
//     ends as "1, 2," and not as "1, 2, ".
//   - One space at the very start of the text is dropped: the tokens of
//     " 4, 5, 6" yield "4, 5, 6".
//   - Tokens are held back as long as their text ends in U+FFFD, which is
//     how the text of a character with bytes still to come ends.
type bpeStreamingDetokenizer struct {
	tokenizer *tokenizer.Tokenizer
	space     int32   // the token that is a single space
	hasSpace  bool    // the vocabulary has such a token
	unflushed []int32 // the tokens whose text has not been returned yet
	started   bool    // some text has been returned
}

func newBPEStreamingDetokenizer(t *tokenizer.Tokenizer) *bpeStreamingDetokenizer {
	d := &bpeStreamingDetokenizer{tokenizer: t}
	d.space, d.hasSpace = t.TokenID("Ġ") // a space in the byte-level alphabet
	return d
}

// Add consumes one token and returns the text that is now complete:
// add_token of the reference.
func (d *bpeStreamingDetokenizer) Add(token int32) string {
	d.unflushed = append(d.unflushed, token)
	text := d.tokenizer.Decode(d.unflushed)
	if strings.HasSuffix(text, "\ufffd") || (d.hasSpace && token == d.space) {
		return ""
	}
	return d.flush(text)
}

// Flush returns the text still held back once generation has ended: finalize
// of the reference.
func (d *bpeStreamingDetokenizer) Flush() string {
	return d.flush(d.tokenizer.Decode(d.unflushed))
}

// flush returns text, which is that of the unflushed tokens, without the
// space the reference trims: _maybe_trim_space.
func (d *bpeStreamingDetokenizer) flush(text string) string {
	d.unflushed = d.unflushed[:0]
	if !d.started {
		text = strings.TrimPrefix(text, " ")
		d.started = text != ""
	}
	return text
}

// serverTemplate makes a chat template the server.Templater.
type serverTemplate struct{ template *chattemplate.Template }

var _ server.Templater = serverTemplate{}

func (t serverTemplate) Render(messages, tools, kwargs json.RawMessage, addGenerationPrompt bool) (string, error) {
	return t.template.Render(chattemplate.Input{
		Messages:            messages,
		Tools:               tools,
		Kwargs:              kwargs,
		AddGenerationPrompt: addGenerationPrompt,
	})
}
