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
	"text/tabwriter"
)

// logLevels are the values --log-level accepts, as in mlx_lm.server.
var logLevels = []string{"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

// flagSet is the flag set of the server together with what parsing fills.
type flagSet struct {
	*flag.FlagSet
	cfg          Config
	templateArgs string  // value of --chat-template-args
	unsupported  *string // the first unsupported flag that was given, or ""
}

// newFlagSet defines the flags of mlx_lm.server with their defaults.
func newFlagSet() *flagSet {
	f := &flagSet{
		FlagSet: flag.NewFlagSet("mlx-lm-go", flag.ContinueOnError),
		cfg: Config{
			Host:            "127.0.0.1",
			Port:            8080,
			LogLevel:        "INFO",
			TopP:            1.0,
			MaxTokens:       512,
			PrefillStepSize: 2048,
		},
		templateArgs: "{}",
	}
	fs, cfg := f.FlagSet, &f.cfg
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
	fs.StringVar(&f.templateArgs, "chat-template-args", f.templateArgs, "JSON object of default arguments for the chat template")
	fs.Var((*float32Value)(&cfg.Temperature), "temp", "default sampling temperature")
	fs.Var((*float32Value)(&cfg.TopP), "top-p", "default nucleus sampling top-p")
	fs.IntVar(&cfg.TopK, "top-k", cfg.TopK, "default top-k sampling, 0 disables it")
	fs.Var((*float32Value)(&cfg.MinP), "min-p", "default min-p sampling, 0 disables it")
	fs.IntVar(&cfg.MaxTokens, "max-tokens", cfg.MaxTokens, "default maximum number of tokens to generate")
	fs.IntVar(&cfg.PrefillStepSize, "prefill-step-size", cfg.PrefillStepSize, "number of prompt tokens processed per step")
	f.unsupported = rejectUnsupported(fs)
	return f
}

// ParseFlags parses mlx_lm.server-compatible flags (without the program name).
// It returns flag.ErrHelp when the arguments ask for help.
func ParseFlags(args []string) (Config, error) {
	fs := newFlagSet()
	if err := fs.Parse(args); err != nil {
		return Config{}, err
	}
	cfg, unsupported := fs.cfg, fs.unsupported
	cfg.ChatTemplateArgs = json.RawMessage(fs.templateArgs)
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

// Usage writes the flags ParseFlags accepts to w, each with its default if
// it has one, and then the flags of mlx_lm.server that are not supported.
func Usage(w io.Writer) {
	var unsupported []string
	table := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(table, "Flags:")
	newFlagSet().VisitAll(func(f *flag.Flag) {
		if slices.Contains(unsupportedValueFlags, f.Name) || slices.Contains(unsupportedSwitches, f.Name) {
			unsupported = append(unsupported, f.Name)
			return
		}
		line := "  --" + f.Name + "\t" + f.Usage
		if f.DefValue != "" {
			line += " (default " + f.DefValue + ")"
		}
		fmt.Fprintln(table, line)
	})
	table.Flush()

	fmt.Fprintln(w, "\nNot supported yet (flags of mlx_lm.server):")
	for _, name := range unsupported {
		fmt.Fprintln(w, "  --"+name)
	}
}

// float32Value is a float32 flag. Its current value is the flag's default.
type float32Value float32

func (f *float32Value) Set(v string) error {
	parsed, err := strconv.ParseFloat(v, 32)
	if err != nil {
		return err
	}
	*f = float32Value(parsed)
	return nil
}

func (f *float32Value) String() string {
	return strconv.FormatFloat(float64(*f), 'g', -1, 32)
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
