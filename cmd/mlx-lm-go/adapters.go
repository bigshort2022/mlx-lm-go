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

func (t serverTokenizer) NewDetokenizer() server.Detokenizer { return newDetokenizer(t.Tokenizer) }

// detokenizer turns the tokens of one generation into text in the pieces of
// the reference's BPEStreamingDetokenizer, which differ from a plain decode in
// two ways. The token that is a single space is held back until the token
// after it, so that " 2" arrives as one piece; the server drops whole pieces
// when it meets a stop string. And one space at the very start of the text is
// dropped.
type detokenizer struct {
	stream   *tokenizer.Stream
	space    int32  // the token that is a single space
	hasSpace bool   // the vocabulary has such a token
	held     string // decoded text not returned yet
	started  bool   // some text has been returned
}

func newDetokenizer(t *tokenizer.Tokenizer) *detokenizer {
	d := &detokenizer{stream: t.NewStream()}
	d.space, d.hasSpace = t.TokenID("Ġ") // a space in the byte-level alphabet
	return d
}

// Add consumes one token and returns the text that is now complete.
func (d *detokenizer) Add(token int32) string {
	d.held += d.stream.Add(token)
	if d.hasSpace && token == d.space {
		return ""
	}
	return d.release()
}

// Flush returns the text still held back once generation has ended.
func (d *detokenizer) Flush() string {
	d.held += d.stream.Flush()
	return d.release()
}

func (d *detokenizer) release() string {
	text := d.held
	d.held = ""
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
