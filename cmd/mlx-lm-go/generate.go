package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"strconv"
	"text/tabwriter"
	"time"

	"github.com/bigshort2022/mlx-lm-go/internal/chattemplate"
	"github.com/bigshort2022/mlx-lm-go/internal/engine"
	"github.com/bigshort2022/mlx-lm-go/internal/generate"
)

// generateFlags are the settings of the generate command.
type generateFlags struct {
	model        string
	prompt       string
	system       string // "" means no system message
	maxTokens    int
	temperature  float32
	templateArgs string // a JSON object of extra template variables
}

// generateFlagSet defines the flags of the generate command, which fill f.
func generateFlagSet(f *generateFlags) *flag.FlagSet {
	fs := flag.NewFlagSet("generate", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	fs.StringVar(&f.model, "model", "", "path or Hugging Face repository of the model")
	fs.StringVar(&f.prompt, "prompt", "", "the user message")
	fs.StringVar(&f.system, "system", "", "the system message")
	fs.IntVar(&f.maxTokens, "max-tokens", 100, "maximum number of tokens to generate")
	fs.Func("temp", "sampling temperature; 0, the default, picks the likeliest token", func(v string) error {
		temperature, err := strconv.ParseFloat(v, 32)
		f.temperature = float32(temperature)
		return err
	})
	fs.StringVar(&f.templateArgs, "chat-template-args", "{}", "JSON object of arguments for the chat template")
	return fs
}

// parseGenerateFlags parses the flags of the generate command. It returns
// flag.ErrHelp when the arguments ask for help.
func parseGenerateFlags(args []string) (generateFlags, error) {
	var f generateFlags
	fs := generateFlagSet(&f)
	if err := fs.Parse(args); err != nil {
		return generateFlags{}, err
	}
	var object map[string]json.RawMessage
	switch {
	case fs.NArg() > 0:
		return generateFlags{}, fmt.Errorf("unexpected argument %q", fs.Arg(0))
	case f.model == "":
		return generateFlags{}, errors.New("--model is required")
	case f.prompt == "":
		return generateFlags{}, errors.New("--prompt is required")
	case f.temperature < 0:
		return generateFlags{}, errors.New("--temp must not be negative")
	case json.Unmarshal([]byte(f.templateArgs), &object) != nil || object == nil:
		return generateFlags{}, errors.New("--chat-template-args must be a JSON object")
	}
	return f, nil
}

// generateUsage writes what generate --help prints: the flags, each with its
// default if it has one.
func generateUsage(w io.Writer) {
	fmt.Fprint(w, "usage: mlx-lm-go generate --model <dir or repo id> --prompt <text> [flags]\n\nFlags:\n")
	table := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	generateFlagSet(new(generateFlags)).VisitAll(func(f *flag.Flag) {
		line := "  --" + f.Name + "\t" + f.Usage
		if f.DefValue != "" {
			line += " (default " + f.DefValue + ")"
		}
		fmt.Fprintln(table, line)
	})
	table.Flush()
}

// runGenerate answers one prompt on stdout and reports the token counts and
// the generation speed on stderr.
func runGenerate(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	f, err := parseGenerateFlags(args)
	if errors.Is(err, flag.ErrHelp) {
		generateUsage(stdout)
		return exitOK
	}
	if err != nil {
		return usageError(stderr, "generate: "+err.Error())
	}
	if err := generateText(ctx, f, stdout, stderr); err != nil {
		return failure(stderr, err)
	}
	return exitOK
}

// chatMessage is one message of the conversation handed to the template.
type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// generateText renders the conversation with the model's chat template and
// writes the model's answer to stdout piece by piece, then a newline.
func generateText(ctx context.Context, f generateFlags, stdout, stderr io.Writer) error {
	m, err := openModel(f.model, "")
	if err != nil {
		return err
	}
	var chat []chatMessage
	if f.system != "" {
		chat = append(chat, chatMessage{Role: "system", Content: f.system})
	}
	chat = append(chat, chatMessage{Role: "user", Content: f.prompt})
	messages, err := json.Marshal(chat)
	if err != nil {
		return err
	}
	prompt, err := m.template.Render(chattemplate.Input{
		Messages:            messages,
		Kwargs:              json.RawMessage(f.templateArgs),
		AddGenerationPrompt: true,
	})
	if err != nil {
		return err
	}
	eng, err := m.loadEngine(generate.Options{})
	if err != nil {
		return err
	}
	defer eng.Close()

	// A failed write to stdout ends the generation with that error.
	ctx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	write := func(text string) {
		if _, err := io.WriteString(stdout, text); err != nil {
			cancel(err)
		}
	}
	detok := newBPEStreamingDetokenizer(m.tokenizer)
	req := engine.Request{
		Prompt:    m.tokenizer.Encode(prompt),
		MaxTokens: f.maxTokens,
		Sampling:  engine.Sampling{Temperature: f.temperature},
	}
	// Like the reference, the rate leaves out the time to the first token,
	// which is that of processing the prompt.
	began := time.Now()
	sawToken := false
	res, err := eng.Generate(ctx, req, func(token int32) {
		if !sawToken {
			began, sawToken = time.Now(), true
		}
		write(detok.Add(token))
	})
	elapsed := time.Since(began)
	write(detok.Flush() + "\n")
	if cause := context.Cause(ctx); cause != nil {
		return cause
	}
	if err != nil {
		return err
	}
	fmt.Fprintf(stderr, "Prompt: %d tokens\nGeneration: %d tokens, %.1f tokens-per-sec\n",
		res.PromptTokens, res.CompletionTokens, float64(res.CompletionTokens)/elapsed.Seconds())
	return nil
}
