package server

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"slices"
	"strconv"
	"strings"
)

// logLevels are the values --log-level accepts, as in mlx_lm.server.
var logLevels = []string{"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

// ParseFlags parses mlx_lm.server-compatible flags (without the program name).
func ParseFlags(args []string) (Config, error) {
	cfg := Config{
		Host:            "127.0.0.1",
		Port:            8080,
		LogLevel:        "INFO",
		TopP:            1.0,
		MaxTokens:       512,
		PrefillStepSize: 2048,
	}
	templateArgs := "{}"

	fs := flag.NewFlagSet("mlx-lm-go", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	fs.StringVar(&cfg.Model, "model", "", "path or Hugging Face repository of the model to serve")
	fs.StringVar(&cfg.Host, "host", cfg.Host, "host to listen on")
	fs.IntVar(&cfg.Port, "port", cfg.Port, "port to listen on")
	fs.Func("allowed-origins", "comma-separated browser origins allowed to call the server, or *", func(v string) error {
		cfg.AllowedOrigins = splitList(v)
		return nil
	})
	fs.StringVar(&cfg.LogLevel, "log-level", cfg.LogLevel, "logging level: "+strings.Join(logLevels, ", "))
	fs.StringVar(&cfg.ChatTemplate, "chat-template", "", "chat template to use instead of the model's")
	fs.StringVar(&templateArgs, "chat-template-args", templateArgs, "JSON object of default arguments for the chat template")
	float32Var(fs, &cfg.Temperature, "temp", "default sampling temperature")
	float32Var(fs, &cfg.TopP, "top-p", "default nucleus sampling top-p")
	fs.IntVar(&cfg.TopK, "top-k", cfg.TopK, "default top-k sampling, 0 disables it")
	float32Var(fs, &cfg.MinP, "min-p", "default min-p sampling, 0 disables it")
	fs.IntVar(&cfg.MaxTokens, "max-tokens", cfg.MaxTokens, "default maximum number of tokens to generate")
	fs.IntVar(&cfg.PrefillStepSize, "prefill-step-size", cfg.PrefillStepSize, "number of prompt tokens processed per step")
	unsupported := rejectUnsupported(fs)

	if err := fs.Parse(args); err != nil {
		return Config{}, err
	}
	cfg.ChatTemplateArgs = json.RawMessage(templateArgs)
	switch {
	case *unsupported != "":
		return Config{}, fmt.Errorf("--%s is not supported yet", *unsupported)
	case fs.NArg() > 0:
		return Config{}, fmt.Errorf("unexpected argument %q", fs.Arg(0))
	case cfg.Model == "":
		return Config{}, errors.New("--model is required")
	case !slices.Contains(logLevels, cfg.LogLevel):
		return Config{}, fmt.Errorf("--log-level must be one of %s", strings.Join(logLevels, ", "))
	case !isJSONObject(cfg.ChatTemplateArgs):
		return Config{}, errors.New("--chat-template-args must be a JSON object")
	}
	return cfg, nil
}

// splitList splits a comma-separated flag value, dropping blank entries.
func splitList(v string) []string {
	var out []string
	for item := range strings.SplitSeq(v, ",") {
		if item = strings.TrimSpace(item); item != "" {
			out = append(out, item)
		}
	}
	return out
}

// isJSONObject reports whether raw is a JSON object.
func isJSONObject(raw json.RawMessage) bool {
	var object map[string]json.RawMessage
	return json.Unmarshal(raw, &object) == nil && object != nil
}

// float32Var registers a flag that parses its value into *dst, which keeps its
// current value as the default.
func float32Var(fs *flag.FlagSet, dst *float32, name, usage string) {
	fs.Func(name, usage, func(v string) error {
		f, err := strconv.ParseFloat(v, 32)
		if err != nil {
			return err
		}
		*dst = float32(f)
		return nil
	})
}

// Flags of mlx_lm.server that this server recognises only to reject them with
// a clear message. Switches take no value.
var (
	unsupportedValueFlags = []string{
		"adapter-path", "draft-model", "num-draft-tokens", "decode-concurrency",
		"prompt-concurrency", "prompt-cache-size", "prompt-cache-bytes",
	}
	unsupportedSwitches = []string{"trust-remote-code", "use-default-chat-template", "pipeline"}
)

// rejectUnsupported registers the unsupported flags on fs. After fs.Parse the
// returned string names the first one that was given, or is empty.
func rejectUnsupported(fs *flag.FlagSet) *string {
	var seen string
	record := func(name string) func(string) error {
		return func(string) error {
			if seen == "" {
				seen = name
			}
			return nil
		}
	}
	for _, name := range unsupportedValueFlags {
		fs.Func(name, "not supported yet", record(name))
	}
	for _, name := range unsupportedSwitches {
		fs.BoolFunc(name, "not supported yet", record(name))
	}
	return &seen
}
