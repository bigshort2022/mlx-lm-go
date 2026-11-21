package server

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func TestParseFlagsDefaults(t *testing.T) {
	got, err := ParseFlags([]string{"--model", "some/model"})
	if err != nil {
		t.Fatalf("ParseFlags: %v", err)
	}
	want := Config{
		Host:             "127.0.0.1",
		Port:             8080,
		Model:            "some/model",
		LogLevel:         "INFO",
		ChatTemplateArgs: json.RawMessage(`{}`),
		Temperature:      0.0,
		TopP:             1.0,
		MaxTokens:        512,
		PrefillStepSize:  2048,
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ParseFlags defaults =\n%+v, want\n%+v", got, want)
	}
}

func TestParseFlagsSetsEachField(t *testing.T) {
	cases := []struct {
		args []string
		get  func(Config) any
		want any
	}{
		{[]string{"--model", "org/name"}, func(c Config) any { return c.Model }, "org/name"},
		{[]string{"--host", "0.0.0.0"}, func(c Config) any { return c.Host }, "0.0.0.0"},
		{[]string{"--port", "9001"}, func(c Config) any { return c.Port }, 9001},
		{[]string{"--allowed-origins", "https://unite4ai.com"}, func(c Config) any { return c.AllowedOrigins }, []string{"https://unite4ai.com"}},
		{[]string{"--allowed-origins", "a,b"}, func(c Config) any { return c.AllowedOrigins }, []string{"a", "b"}},
		{[]string{"--allowed-origins", "*"}, func(c Config) any { return c.AllowedOrigins }, []string{"*"}},
		{[]string{"--log-level", "DEBUG"}, func(c Config) any { return c.LogLevel }, "DEBUG"},
		{[]string{"--chat-template", "{{ messages }}"}, func(c Config) any { return c.ChatTemplate }, "{{ messages }}"},
		{[]string{"--chat-template-args", `{"enable_thinking":false}`}, func(c Config) any { return string(c.ChatTemplateArgs) }, `{"enable_thinking":false}`},
		{[]string{"--temp", "0.7"}, func(c Config) any { return c.Temperature }, float32(0.7)},
		{[]string{"--top-p", "0.9"}, func(c Config) any { return c.TopP }, float32(0.9)},
		{[]string{"--top-k", "40"}, func(c Config) any { return c.TopK }, 40},
		{[]string{"--min-p", "0.05"}, func(c Config) any { return c.MinP }, float32(0.05)},
		{[]string{"--max-tokens", "64"}, func(c Config) any { return c.MaxTokens }, 64},
		{[]string{"--prefill-step-size", "512"}, func(c Config) any { return c.PrefillStepSize }, 512},
		{[]string{"--port=9002"}, func(c Config) any { return c.Port }, 9002},
	}
	for _, tc := range cases {
		t.Run(tc.args[0], func(t *testing.T) {
			cfg, err := ParseFlags(append([]string{"--model", "m"}, tc.args...))
			if err != nil {
				t.Fatalf("ParseFlags(%q): %v", tc.args, err)
			}
			if got := tc.get(cfg); !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("ParseFlags(%q) set %#v, want %#v", tc.args, got, tc.want)
			}
		})
	}
}

func TestParseFlagsRejectsUnsupportedFlags(t *testing.T) {
	cases := [][]string{
		{"--adapter-path", "adapters"},
		{"--draft-model", "org/draft"},
		{"--num-draft-tokens", "3"},
		{"--trust-remote-code"},
		{"--use-default-chat-template"},
		{"--decode-concurrency", "32"},
		{"--prompt-concurrency", "8"},
		{"--pipeline"},
		{"--prompt-cache-size", "10"},
		{"--prompt-cache-bytes", "1GB"},
	}
	for _, args := range cases {
		t.Run(args[0], func(t *testing.T) {
			_, err := ParseFlags(append([]string{"--model", "m"}, args...))
			if err == nil {
				t.Fatalf("ParseFlags(%q) succeeded, want an error", args)
			}
			for _, want := range []string{args[0], "not supported yet"} {
				if !strings.Contains(err.Error(), want) {
					t.Errorf("error %q does not contain %q", err, want)
				}
			}
		})
	}
}

func TestParseFlagsErrors(t *testing.T) {
	cases := []struct {
		name string
		args []string
		want string // substring of the error
	}{
		{"unknown flag", []string{"--model", "m", "--nope", "1"}, "nope"},
		{"missing model", []string{"--port", "9001"}, "--model"},
		{"empty model", []string{"--model", ""}, "--model"},
		{"template args not JSON", []string{"--model", "m", "--chat-template-args", "{not json"}, "--chat-template-args"},
		{"template args a list", []string{"--model", "m", "--chat-template-args", "[1, 2]"}, "--chat-template-args"},
		{"template args null", []string{"--model", "m", "--chat-template-args", "null"}, "--chat-template-args"},
		{"unknown log level", []string{"--model", "m", "--log-level", "LOUD"}, "--log-level"},
		{"port not a number", []string{"--model", "m", "--port", "http"}, "port"},
		{"stray argument", []string{"--model", "m", "serve"}, "serve"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ParseFlags(tc.args)
			if err == nil {
				t.Fatalf("ParseFlags(%q) succeeded, want an error", tc.args)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error %q does not contain %q", err, tc.want)
			}
		})
	}
}

func TestParseFlagsAllowedOriginsIgnoresEmptyEntries(t *testing.T) {
	cfg, err := ParseFlags([]string{"--model", "m", "--allowed-origins", " https://a.example ,, https://b.example"})
	if err != nil {
		t.Fatalf("ParseFlags: %v", err)
	}
	if want := []string{"https://a.example", "https://b.example"}; !reflect.DeepEqual(cfg.AllowedOrigins, want) {
		t.Fatalf("AllowedOrigins = %q, want %q", cfg.AllowedOrigins, want)
	}
}

// usageSection returns the lines of one section of Usage, which begins with
// the given heading and ends with a blank line or the end of the text.
func usageSection(t *testing.T, usage, heading string) []string {
	t.Helper()
	_, rest, found := strings.Cut(usage, heading+"\n")
	if !found {
		t.Fatalf("Usage has no line %q:\n%s", heading, usage)
	}
	section, _, _ := strings.Cut(rest, "\n\n")
	return strings.Split(strings.TrimSuffix(section, "\n"), "\n")
}

func TestUsageListsEverySupportedFlagWithItsDefault(t *testing.T) {
	var out strings.Builder
	Usage(&out)

	defaults := map[string]string{ // "" means the flag has no default
		"--allowed-origins":    "",
		"--chat-template":      "",
		"--chat-template-args": "{}",
		"--host":               "127.0.0.1",
		"--log-level":          "INFO",
		"--max-tokens":         "512",
		"--min-p":              "0",
		"--model":              "",
		"--port":               "8080",
		"--prefill-step-size":  "2048",
		"--temp":               "0",
		"--top-k":              "0",
		"--top-p":              "1",
	}
	lines := usageSection(t, out.String(), "Flags:")
	if len(lines) != len(defaults) {
		t.Errorf("Usage lists %d flags, want %d:\n%s", len(lines), len(defaults), out.String())
	}
	for _, line := range lines {
		fields := strings.Fields(line)
		if len(fields) < 2 {
			t.Errorf("line %q does not describe its flag", line)
			continue
		}
		name := fields[0]
		def, ok := defaults[name]
		switch {
		case !ok:
			t.Errorf("Usage lists %s, which is not a supported flag", name)
		case def == "" && strings.Contains(line, "(default"):
			t.Errorf("line %q gives a default, want none", line)
		case def != "" && !strings.HasSuffix(line, "(default "+def+")"):
			t.Errorf("line %q does not end with (default %s)", line, def)
		}
		delete(defaults, name)
	}
	for name := range defaults {
		t.Errorf("Usage does not list %s", name)
	}
}

func TestUsageListsTheUnsupportedFlagsApart(t *testing.T) {
	var out strings.Builder
	Usage(&out)

	want := []string{
		"--adapter-path", "--decode-concurrency", "--draft-model", "--num-draft-tokens", "--pipeline",
		"--prompt-cache-bytes", "--prompt-cache-size", "--prompt-concurrency", "--trust-remote-code",
		"--use-default-chat-template",
	}
	var got []string
	for _, line := range usageSection(t, out.String(), "Not supported yet (flags of mlx_lm.server):") {
		got = append(got, strings.TrimSpace(line))
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("unsupported flags = %q, want %q", got, want)
	}
}
