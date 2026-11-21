package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/bigshort2022/mlx-lm-go/internal/hub"
)

// Exercise the real command in a subprocess: TestMain otherwise shares an
// engine between the lightweight Qwen2 tests. The opt-in gate also prevents
// concurrent default go test packages loading the 27B checkpoint.
func TestQwen38Binary(t *testing.T) {
	if os.Getenv("MLX_LM_GO_TEST_QWEN38") != "1" {
		t.Skip("set MLX_LM_GO_TEST_QWEN38=1 to run the pinned checkpoint gate")
	}
	binary := filepath.Join(t.TempDir(), "mlx-lm-go")
	if out, err := exec.CommandContext(t.Context(), "go", "build", "-o", binary, ".").CombinedOutput(); err != nil {
		t.Fatalf("go build: %v\n%s", err, out)
	}
	for _, tc := range []struct{ name, repo, revision string }{
		{"qwen38", "mlx-community/Qwen3.8-27B-4bit", "10c35caafbb80f7dc6a7a432cdd11af10a6d4818"},
		{"qwen35", "mlx-community/Qwen3.5-9B-MLX-4bit", "938d8919941c6e7efd3c7150eff7fe9d12afa631"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cached, err := hub.Resolve(tc.repo)
			if err != nil {
				t.Fatal(err)
			}
			model := filepath.Join(filepath.Dir(cached), tc.revision)
			if _, err := os.Stat(filepath.Join(model, "config.json")); err != nil {
				t.Fatalf("pinned snapshot %s@%s: %v", tc.repo, tc.revision, err)
			}
			t.Run("cli", func(t *testing.T) {
				cmd := exec.CommandContext(t.Context(), binary, "generate", "--model", model,
					"--prompt", "What is 17 times 23? Reply with only the integer.",
					"--chat-template-args", `{"enable_thinking":false}`, "--max-tokens", "48")
				var stderr bytes.Buffer
				cmd.Stderr = &stderr
				out, err := cmd.Output()
				if err != nil {
					t.Fatalf("generate: %v\n%s", err, &stderr)
				}
				if got := string(out); got != "391\n" {
					t.Errorf("generate output = %q, want %q", got, "391\n")
				}
			})
			t.Run("http", func(t *testing.T) {
				checkQwenHTTP(t, binary, model, tc.name == "qwen38")
			})
		})
	}
}

func checkQwenHTTP(t *testing.T, binary, model string, flagship bool) {
	t.Helper()
	s := startServing(t, func(ctx context.Context, stderr io.Writer) int {
		cmd := exec.CommandContext(ctx, binary, "server", "--model", model, "--port", "0", "--prefill-step-size", "8")
		cmd.Stderr = stderr
		cmd.Cancel = func() error { return cmd.Process.Signal(syscall.SIGTERM) }
		cmd.WaitDelay = 5 * time.Second
		if err := cmd.Run(); err != nil && (ctx.Err() == nil || cmd.ProcessState == nil || !cmd.ProcessState.Success()) {
			return failure(stderr, err)
		}
		return exitOK
	})

	const chatPath = "/v1/chat/completions"
	const question = `{"messages":[{"role":"user","content":"What is 17 times 23? Reply with only the integer."}],"max_tokens":48`
	body := question + `,"chat_template_kwargs":{"enable_thinking":false}}`
	got := decodeCompletion(t, s.post(t, chatPath, body))
	if c := got.Choices[0]; c.Message.Content != "391" || c.FinishReason != "stop" {
		t.Fatalf("non-thinking chat = %+v, want 391 and stop", c)
	}
	if !flagship {
		return
	}
	// Pinned Python mlx-lm 0.32.2, using the same 8-token prefill and
	// --prompt-cache-size 0: 28 prompt tokens, 4 completion tokens.
	if got.Usage.PromptTokens != 28 || got.Usage.CompletionTokens != 4 {
		t.Fatalf("non-thinking usage = %+v, want 28 prompt and 4 completion tokens", got.Usage)
	}

	status, modelJSON := s.get(t, "/v1/models")
	var models struct {
		Data []struct{ ID string }
	}
	if status != http.StatusOK || json.Unmarshal([]byte(modelJSON), &models) != nil ||
		len(models.Data) != 1 || models.Data[0].ID != model {
		t.Fatalf("GET /v1/models = %d %s, want only %q", status, modelJSON, model)
	}

	stream := s.post(t, chatPath, question+`,"chat_template_kwargs":{"enable_thinking":false},"stream":true,"stream_options":{"include_usage":true}}`)
	if decoded := decodeStream(t, stream); strings.Join(decoded.Pieces, "") != "391" || decoded.FinishReason != "stop" {
		t.Fatalf("non-thinking stream = %+v, want 391 and stop", decoded)
	}
	var usage completion
	for line := range strings.Lines(stream) {
		data, ok := strings.CutPrefix(strings.TrimSpace(line), "data: ")
		if !ok || data == "[DONE]" {
			continue
		}
		var event completion
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			t.Fatal(err)
		}
		if event.Usage.CompletionTokens > 0 {
			usage = event
		}
	}
	if len(usage.Choices) != 0 || usage.Usage != got.Usage || usage.Usage.CompletionTokens == 0 {
		t.Fatalf("stream usage = %+v, want buffered usage %+v and no choices", usage, got.Usage)
	}

	thinking := s.post(t, chatPath, question+`,"chat_template_kwargs":{"enable_thinking":true}}`)
	thinkingCompletion := decodeCompletion(t, thinking)
	if thinkingCompletion.Choices[0].FinishReason != "length" ||
		thinkingCompletion.Usage.PromptTokens != 68 || thinkingCompletion.Usage.CompletionTokens != 48 {
		t.Fatalf("thinking completion = %+v, want Python's length stop with usage 68/48", thinkingCompletion)
	}
	var thought struct {
		Choices []struct {
			Message struct {
				Content   json.RawMessage
				Reasoning string
			}
		}
	}
	if err := json.Unmarshal([]byte(thinking), &thought); err != nil {
		t.Fatal(err)
	}
	if len(thought.Choices) != 1 || string(thought.Choices[0].Message.Content) != "null" ||
		strings.TrimSpace(thought.Choices[0].Message.Reasoning) == "" ||
		strings.Contains(thinking, "<think>") || strings.Contains(thinking, "</think>") {
		t.Fatalf("thinking reply does not send reasoning with null content: %s", thinking)
	}

	raw := decodeCompletion(t, s.post(t, "/v1/completions", `{"prompt":"1, 2, 3, 4,","max_tokens":6}`))
	if raw.Choices[0].Text == "" {
		t.Fatal("raw completion returned no text")
	}
	for _, media := range []string{"image_url", "video_url"} {
		body := fmt.Sprintf(`{"messages":[{"role":"user","content":[{"type":%q}]}]}`, media)
		resp, err := http.Post(s.url+chatPath, "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusBadRequest {
			t.Errorf("chat with %s returned %d, want 400", media, resp.StatusCode)
		}
	}
}
