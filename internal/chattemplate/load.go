package chattemplate

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
)

const (
	templateFile        = "chat_template.jinja"
	tokenizerConfigFile = "tokenizer_config.json"
)

// specialTokenNames are the tokenizer_config.json entries a template can read.
var specialTokenNames = []string{"bos_token", "eos_token", "pad_token", "unk_token"}

// Load reads the chat template from dir: chat_template.jinja if present,
// otherwise "chat_template" in tokenizer_config.json. Special-token strings
// (bos_token, eos_token, pad_token, unk_token) come from tokenizer_config.json
// and are available to the template under those names.
func Load(dir string) (*Template, error) {
	tokenizerConfig, err := readTokenizerConfig(dir)
	if err != nil {
		return nil, err
	}
	specialTokens := map[string]string{}
	for _, name := range specialTokenNames {
		if token, ok := specialToken(tokenizerConfig[name]); ok {
			specialTokens[name] = token
		}
	}

	source, err := os.ReadFile(filepath.Join(dir, templateFile))
	if err == nil {
		return Parse(string(source), specialTokens)
	}
	if !errors.Is(err, fs.ErrNotExist) {
		return nil, err
	}
	var inline string
	if err := json.Unmarshal(tokenizerConfig["chat_template"], &inline); err != nil || inline == "" {
		return nil, fmt.Errorf("no chat template in %s: neither %s nor a \"chat_template\" string in %s",
			dir, templateFile, tokenizerConfigFile)
	}
	return Parse(inline, specialTokens)
}

// readTokenizerConfig returns the top-level entries of tokenizer_config.json,
// or none when the model directory has no such file.
func readTokenizerConfig(dir string) (map[string]json.RawMessage, error) {
	raw, err := os.ReadFile(filepath.Join(dir, tokenizerConfigFile))
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var entries map[string]json.RawMessage
	if err := json.Unmarshal(raw, &entries); err != nil {
		return nil, fmt.Errorf("%s: %w", tokenizerConfigFile, err)
	}
	return entries, nil
}

// specialToken extracts a token's text from its tokenizer_config.json entry,
// which is either a string or an object with the text under "content". A
// missing or null entry yields no token, leaving the variable undefined in the
// template, as Python does.
func specialToken(entry json.RawMessage) (string, bool) {
	var value any
	if entry == nil || json.Unmarshal(entry, &value) != nil {
		return "", false
	}
	switch value := value.(type) {
	case string:
		return value, true
	case map[string]any:
		content, ok := value["content"].(string)
		return content, ok
	}
	return "", false
}
