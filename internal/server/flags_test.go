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
