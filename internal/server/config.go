// Package server is the OpenAI-compatible HTTP layer of mlx-lm-go. It is
// shape-compatible with Python mlx_lm.server and safe to reach from a browser.
package server

import (
	"encoding/json"
	"io/fs"

	"github.com/bigshort2022/mlx-lm-go/internal/engine"
)

// Config holds the server settings. ParseFlags fills it from mlx_lm.server
// flags.
type Config struct {
	Host             string          // default "127.0.0.1"
	Port             int             // default 8080; must be the port the listener is bound to, see New
	AllowedOrigins   []string        // default empty: same-origin only. "*" allows any origin.
	Model            string          // value of --model; the id listed by /v1/models
	LogLevel         string          // default "INFO"
	ChatTemplate     string          // --chat-template; "" means use the model's
	ChatTemplateArgs json.RawMessage // default {}
	Temperature      float32         // default 0.0
	TopP             float32         // default 1.0
	TopK             int             // default 0
	MinP             float32         // default 0.0
	MaxTokens        int             // default 512
	PrefillStepSize  int             // default 2048
}

// Tokenizer turns text into token ids and creates detokenizers. The server
// calls it for one request at a time.
type Tokenizer interface {
	// Encode returns the token ids of text.
	Encode(text string) []int32
	// NewDetokenizer returns a detokenizer for one generation.
	NewDetokenizer() Detokenizer
}

// Detokenizer turns a stream of generated tokens back into text.
type Detokenizer interface {
	// Add consumes one token and returns the text that is now complete. It may
	// return "" while it waits for more tokens.
	Add(token int32) string
	// Flush returns the text still held back once generation has ended.
	Flush() string
}

// Templater renders chat messages into the prompt text of the served model.
type Templater interface {
	// Render applies the chat template. messages and tools are the JSON values
	// of the request fields of the same name (tools is JSON null when there
	// are none) and kwargs is a JSON object of extra template variables.
	Render(messages, tools, kwargs json.RawMessage, addGenerationPrompt bool) (string, error)
}

// Deps are the collaborators the server drives. The server uses them for one
// request at a time, so they need not be safe for concurrent use.
type Deps struct {
	Engine    engine.Engine
	Tokenizer Tokenizer
	Template  Templater
	WebUI     fs.FS // nil: no UI is served
}
