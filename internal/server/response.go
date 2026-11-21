package server

import (
	"crypto/rand"
	"fmt"
	"runtime"
	"time"

	"github.com/bigshort2022/mlx-lm-go/internal/engine"
)

// systemFingerprint identifies the server build in every completion response.
var systemFingerprint = "mlx-lm-go-" + runtime.GOOS + "-" + runtime.GOARCH

// completionResponse is the body of a completion and of each stream chunk, for
// both chat and text completions. Field order follows the reference.
type completionResponse struct {
	ID                string   `json:"id"`
	SystemFingerprint string   `json:"system_fingerprint"`
	Object            string   `json:"object"`
	Model             string   `json:"model"`
	Created           int64    `json:"created"`
	Choices           []choice `json:"choices"`
	Usage             *usage   `json:"usage,omitempty"`
}

// choice carries a message (chat), a delta (chat stream) or text (text
// completions). FinishReason is null until the last chunk of a stream.
type choice struct {
	Index        int          `json:"index"`
	FinishReason *string      `json:"finish_reason"`
	Message      *chatMessage `json:"message,omitempty"`
	Delta        *chatMessage `json:"delta,omitempty"`
	Text         *string      `json:"text,omitempty"`
}

// chatMessage is the assistant message of a chat completion. Like the
// reference it leaves out content when there is none.
type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content,omitempty"`
}

// usage is the token accounting of a completion.
type usage struct {
	PromptTokens        int                 `json:"prompt_tokens"`
	CompletionTokens    int                 `json:"completion_tokens"`
	TotalTokens         int                 `json:"total_tokens"`
	PromptTokensDetails promptTokensDetails `json:"prompt_tokens_details"`
}

type promptTokensDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

// usageOf copies the engine's token counts into the response shape.
func usageOf(res engine.Result) *usage {
	return &usage{
		PromptTokens:        res.PromptTokens,
		CompletionTokens:    res.CompletionTokens,
		TotalTokens:         res.PromptTokens + res.CompletionTokens,
		PromptTokensDetails: promptTokensDetails{CachedTokens: res.CachedTokens},
	}
}

// reply builds the response bodies of one completion. Every body of a
// stream carries the same id and creation time.
type reply struct {
	chat    bool // chat completion; otherwise a text completion
	id      string
	model   string
	created int64
}

// newReply starts the response to a completion request for the named model.
func newReply(chat bool, model string) reply {
	prefix := "cmpl-"
	if chat {
		prefix = "chatcmpl-"
	}
	return reply{chat: chat, id: newID(prefix), model: model, created: time.Now().Unix()}
}

// body returns a response with the given object name and choices.
func (p reply) body(object string, choices ...choice) completionResponse {
	return completionResponse{
		ID:                p.id,
		SystemFingerprint: systemFingerprint,
		Object:            object,
		Model:             p.model,
		Created:           p.created,
		Choices:           append([]choice{}, choices...),
	}
}

// full is the body of a completion that is not streamed.
func (p reply) full(text string, res engine.Result) completionResponse {
	object, c := "text_completion", choice{FinishReason: &res.FinishReason, Text: &text}
	if p.chat {
		object, c = "chat.completion", choice{FinishReason: &res.FinishReason, Message: &chatMessage{Role: "assistant", Content: text}}
	}
	body := p.body(object, c)
	body.Usage = usageOf(res)
	return body
}

// chunk is one event of a stream. finish is empty for every event but the
// last, which carries no text.
func (p reply) chunk(text, finish string) completionResponse {
	object, c := "text_completion", choice{Text: &text}
	if p.chat {
		object, c = "chat.completion.chunk", choice{Delta: &chatMessage{Role: "assistant", Content: text}}
	}
	if finish != "" {
		c.FinishReason = &finish
	}
	return p.body(object, c)
}

// usageChunk is the optional event that reports token counts at the end of a
// stream. Like the reference it has no choices.
func (p reply) usageChunk(res engine.Result) completionResponse {
	object := "text_completion"
	if p.chat {
		object = "chat.completion"
	}
	body := p.body(object)
	body.Usage = usageOf(res)
	return body
}

// newID returns prefix followed by a random version 4 UUID.
func newID(prefix string) string {
	var b [16]byte
	rand.Read(b[:])
	b[6] = b[6]&0x0f | 0x40 // version 4
	b[8] = b[8]&0x3f | 0x80 // RFC 9562 variant
	return fmt.Sprintf("%s%x-%x-%x-%x-%x", prefix, b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}
