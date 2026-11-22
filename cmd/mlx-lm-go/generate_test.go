package main

import (
	"bytes"
	"context"
	"errors"
	"regexp"
	"slices"
	"strings"
	"testing"
)

func TestGenerateAnswersLikeTheReference(t *testing.T) {
	needModel(t)

	code, stdout, stderr := runCommand(t, "generate", "--model", referenceModel,
		"--prompt", "Count from 1 to 5.", "--max-tokens", "24")
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr:\n%s", code, stderr)
	}
	if want := "1, 2, 3, 4, 5\n"; stdout != want {
		t.Errorf("stdout = %q, want %q", stdout, want)
	}
}

func TestGenerateWithASystemMessage(t *testing.T) {
	needModel(t)

	code, stdout, stderr := runCommand(t, "generate", "--model", referenceModel,
		"--system", "You are a terse assistant.", "--prompt", "What is the capital of France?")
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr:\n%s", code, stderr)
	}
	if want := "Paris.\n"; stdout != want {
		t.Errorf("stdout = %q, want %q", stdout, want)
	}
}

func TestGenerateReportsTokenCountsAndSpeed(t *testing.T) {
	needModel(t)

	_, _, stderr := runCommand(t, "generate", "--model", referenceModel,
		"--prompt", "Count from 1 to 5.", "--max-tokens", "24")
	// The counts are those of testdata/server/chat.json.
	stats := regexp.MustCompile(`^Prompt: 37 tokens\nGeneration: 14 tokens, \d+\.\d tokens-per-sec\n$`)
	if !stats.MatchString(stderr) {
		t.Errorf("stderr = %q, want it to match %s", stderr, stats)
	}
}

// The template refuses an argument of apply_chat_template among its extra
// variables, which shows that the arguments reach it.
func TestGeneratePassesChatTemplateArgsToTheTemplate(t *testing.T) {
	needModel(t)

	code, stdout, stderr := runCommand(t, "generate", "--model", referenceModel,
		"--prompt", "Hi", "--chat-template-args", `{"messages": 1}`)
	if code != 1 {
		t.Errorf("exit code = %d, want 1", code)
	}
	if stdout != "" {
		t.Errorf("stdout = %q, want nothing", stdout)
	}
	if !strings.Contains(stderr, `kwargs: "messages"`) {
		t.Errorf("stderr does not report the template's refusal:\n%s", stderr)
	}
}

func TestGeneratePassesTheTemperatureToTheEngine(t *testing.T) {
	needModel(t)

	code, _, stderr := runCommand(t, "generate", "--model", referenceModel,
		"--prompt", "Hi", "--temp", "0.7", "--max-tokens", "1")
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr:\n%s", code, stderr)
	}
	if got := shared.lastRequest().Sampling.Temperature; got != 0.7 {
		t.Errorf("the engine sampled with temperature %v, want 0.7", got)
	}
}

// The tokenizer's end-of-sequence token is what stops a model whose
// config.json names none.
func TestGenerateGivesTheEngineTheTokenizersEOSToken(t *testing.T) {
	needModel(t)

	code, _, stderr := runCommand(t, "generate", "--model", referenceModel, "--prompt", "Hi", "--max-tokens", "1")
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr:\n%s", code, stderr)
	}
	const imEnd = 151645 // <|im_end|>, the eos_token of tokenizer_config.json
	if got := shared.lastOptions().EOSTokenIDs; !slices.Equal(got, []int32{imEnd}) {
		t.Errorf("the engine was loaded with the fallback EOS tokens %v, want [%d]", got, imEnd)
	}
}

// brokenPipe is an output that nobody reads any more.
type brokenPipe struct{}

func (brokenPipe) Write([]byte) (int, error) { return 0, errors.New("broken pipe") }

func TestGenerateStopsWhenItsOutputFails(t *testing.T) {
	needModel(t)

	var stderr bytes.Buffer
	code := run(context.Background(), []string{"generate", "--model", referenceModel,
		"--prompt", "Write a long story.", "--max-tokens", "2000"}, brokenPipe{}, &stderr)
	if code != 1 {
		t.Errorf("exit code = %d, want 1", code)
	}
	if got, want := stderr.String(), "mlx-lm-go: broken pipe\n"; got != want {
		t.Errorf("stderr = %q, want %q", got, want)
	}
	if generated := shared.lastResult().CompletionTokens; generated > 2 {
		t.Errorf("the engine generated %d tokens after the output had failed", generated)
	}
}

func TestGenerateReportsAModelThatIsNotCached(t *testing.T) {
	t.Setenv("HF_HUB_CACHE", t.TempDir())

	code, stdout, stderr := runCommand(t, "generate", "--model", "nobody/no-such-model", "--prompt", "Hi")
	if code != 1 {
		t.Errorf("exit code = %d, want 1", code)
	}
	if stdout != "" {
		t.Errorf("stdout = %q, want nothing", stdout)
	}
	if !strings.Contains(stderr, "hf download nobody/no-such-model") {
		t.Errorf("stderr does not say how to get the model:\n%s", stderr)
	}
}

func TestGenerateUsageErrors(t *testing.T) {
	tests := []struct {
		name   string
		args   []string
		reason string
	}{
		{"no model", []string{"--prompt", "Hi"}, "--model is required"},
		{"no prompt", []string{"--model", "m"}, "--prompt is required"},
		{"stray argument", []string{"--model", "m", "--prompt", "Hi", "there"}, `unexpected argument "there"`},
		{"negative temperature", []string{"--model", "m", "--prompt", "Hi", "--temp", "-1"}, "--temp must not be negative"},
		{"template arguments that are not an object", []string{"--model", "m", "--prompt", "Hi", "--chat-template-args", "[1]"},
			"--chat-template-args must be a JSON object"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			code, _, stderr := runCommand(t, append([]string{"generate"}, tt.args...)...)
			if code != 2 {
				t.Errorf("exit code = %d, want 2", code)
			}
			if !strings.Contains(stderr, "generate: "+tt.reason) {
				t.Errorf("stderr does not give the reason %q:\n%s", tt.reason, stderr)
			}
		})
	}
}
