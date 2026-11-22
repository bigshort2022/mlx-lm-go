package main

import (
	"context"
	"encoding/json"
	"io"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestServerPromptCache(t *testing.T) {
	needModel(t)
	binary := filepath.Join(t.TempDir(), "mlx-lm-go")
	if out, err := exec.CommandContext(t.Context(), "go", "build", "-o", binary, ".").CombinedOutput(); err != nil {
		t.Fatalf("go build: %v\n%s", err, out)
	}
	s := startServing(t, func(ctx context.Context, stderr io.Writer) int {
		cmd := exec.CommandContext(ctx, binary, "server", "--model", referenceModel, "--port", "0",
			"--prompt-cache-size", "2", "--prompt-cache-bytes", "128MB")
		cmd.Stderr = stderr
		cmd.Cancel = func() error { return cmd.Process.Signal(syscall.SIGTERM) }
		cmd.WaitDelay = 5 * time.Second
		if err := cmd.Run(); err != nil && (ctx.Err() == nil || cmd.ProcessState == nil || !cmd.ProcessState.Success()) {
			return failure(stderr, err)
		}
		return exitOK
	})
	base := strings.Repeat("The blue car moved slowly down the road. ", 30)
	request := func(prompt string) cacheTestCompletion {
		t.Helper()
		body, err := json.Marshal(map[string]any{"prompt": prompt, "max_tokens": 6, "presence_penalty": 0.5, "presence_context_size": 128})
		if err != nil {
			t.Fatal(err)
		}
		var got cacheTestCompletion
		if err := json.Unmarshal([]byte(s.post(t, "/v1/completions", string(body))), &got); err != nil {
			t.Fatal(err)
		}
		if len(got.Choices) != 1 || got.Usage.Prompt == 0 {
			t.Fatalf("incomplete response: %+v", got)
		}
		return got
	}
	cold := request(base + "The driver said:")
	warm := request(base + "The driver said:")
	if cold.Usage.Details.Cached != 0 || warm.Usage.Details.Cached < cold.Usage.Prompt-32 || warm.Usage.Details.Cached >= warm.Usage.Prompt {
		t.Fatalf("unexpected cold/warm accounting: %+v / %+v", cold.Usage, warm.Usage)
	}
	if cold.Choices[0] != warm.Choices[0] || cold.Usage.Completion != warm.Usage.Completion {
		t.Fatalf("cache changed the completion: %+v / %+v", cold, warm)
	}
	branch := request(base + "The passenger asked a question:")
	if branch.Usage.Details.Cached < cold.Usage.Prompt-32 {
		t.Fatalf("no shared prefix on branch: %+v", branch.Usage)
	}
	if miss := request("Completely unrelated. " + base); miss.Usage.Details.Cached != 0 {
		t.Fatalf("changed leading tokens hit the cache: %+v", miss.Usage)
	}
	for _, prompt := range []string{"Alpha. " + base, "Beta. " + base} {
		if miss := request(prompt); miss.Usage.Details.Cached != 0 {
			t.Fatalf("unrelated prompt hit the cache: %+v", miss.Usage)
		}
	}
	if hit := request("Alpha. " + base); hit.Usage.Details.Cached == 0 {
		t.Fatal("the second most recent prefix was not retained")
	}
	request("Gamma. " + base)
	if miss := request("Beta. " + base); miss.Usage.Details.Cached != 0 {
		t.Fatalf("least recently used prefix was not evicted: %+v", miss.Usage)
	}
}

type cacheTestCompletion struct {
	Choices []struct {
		Text   string `json:"text"`
		Finish string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		Prompt     int `json:"prompt_tokens"`
		Completion int `json:"completion_tokens"`
		Details    struct {
			Cached int `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
	} `json:"usage"`
}
