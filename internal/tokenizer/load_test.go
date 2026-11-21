package tokenizer

import (
	"fmt"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestLoadTinyTokenizer(t *testing.T) {
	tok := loadTiny(t)

	if got := tok.VocabSize(); got != 23 {
		t.Errorf("VocabSize() = %d, want 23 (18 base + 5 added)", got)
	}
	for token, want := range map[string]int32{"abc": 17, "<|im_end|>": 19} {
		got, ok := tok.TokenID(token)
		if !ok || got != want {
			t.Errorf("TokenID(%q) = %d, %v, want %d, true", token, got, ok, want)
		}
	}
	if got, ok := tok.TokenID("nope"); ok {
		t.Errorf("TokenID(%q) = %d, true, want not found", "nope", got)
	}
}

func TestLoadMissingDirectoryFails(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "missing")
	_, err := Load(dir)
	if err == nil {
		t.Fatalf("Load(%q) succeeded, want an error", dir)
	}
	if !strings.Contains(err.Error(), "tokenizer.json") {
		t.Errorf("Load(%q) error = %q, want it to mention tokenizer.json", dir, err)
	}
}

func TestLoadMatchesMeta(t *testing.T) {
	for _, family := range families {
		t.Run(family, func(t *testing.T) {
			tok := loadFamily(t, family)
			meta := readGoldenMeta(t, family)

			if got := tok.VocabSize(); got != meta.VocabSize {
				t.Errorf("VocabSize() = %d, want %d", got, meta.VocabSize)
			}
			token, id, ok := tok.EOSToken()
			want := meta.SpecialTokens
			if !ok || token != want.EOSToken || id != want.EOSTokenID {
				t.Errorf("EOSToken() = %q, %d, %v, want %q, %d, true", token, id, ok, want.EOSToken, want.EOSTokenID)
			}
		})
	}
}

func TestLoadWithoutConfigHasNoEOSToken(t *testing.T) {
	tok := loadTiny(t) // testdata/tiny has no tokenizer_config.json

	if token, id, ok := tok.EOSToken(); ok {
		t.Errorf("EOSToken() = %q, %d, true, want not ok", token, id)
	}
}

// tokenizer_config.json names the EOS token either as a string or as a
// serialized AddedToken, an object whose "content" is the token.
func TestEOSTokenFromConfig(t *testing.T) {
	tests := []struct {
		name      string
		config    string
		wantToken string
		wantID    int32
		wantOK    bool
	}{
		{"named", `{"eos_token": "<|im_end|>"}`, "<|im_end|>", 19, true},
		{"object", `{"eos_token": {"content": "<|im_end|>"}}`, "<|im_end|>", 19, true},
		{
			"serialized AddedToken",
			`{"eos_token": {"__type": "AddedToken", "content": "<|im_end|>", "lstrip": false, "normalized": false, "rstrip": false, "single_word": false}}`,
			"<|im_end|>", 19, true,
		},
		{"null", `{"eos_token": null}`, "", 0, false},
		{"absent", `{}`, "", 0, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := tinyWithConfig(t, tt.config)
			tok, err := Load(dir)
			if err != nil {
				t.Fatalf("Load error: %v", err)
			}
			token, id, ok := tok.EOSToken()
			if token != tt.wantToken || id != tt.wantID || ok != tt.wantOK {
				t.Errorf("EOSToken() = %q, %d, %v, want %q, %d, %v", token, id, ok, tt.wantToken, tt.wantID, tt.wantOK)
			}
		})
	}
}

func TestLoadRejectsEOSTokenOutsideVocabulary(t *testing.T) {
	for _, config := range []string{`{"eos_token": "</s>"}`, `{"eos_token": {"content": "</s>"}}`} {
		dir := tinyWithConfig(t, config)
		_, err := Load(dir)
		if err == nil {
			t.Fatalf("Load with %s succeeded, want an error", config)
		}
		for _, want := range []string{"eos_token", `"</s>"`} {
			if !strings.Contains(err.Error(), want) {
				t.Errorf("Load with %s: error = %q, want it to mention %s", config, err, want)
			}
		}
	}
}

func TestLoadRejectsEOSTokenOfAnotherType(t *testing.T) {
	tests := []struct {
		name     string
		eosToken string // JSON value of eos_token
	}{
		{"number", `42`},
		{"boolean", `true`},
		{"array", `["<|im_end|>"]`},
		{"object without content", `{"lstrip": false}`},
		{"object with null content", `{"content": null}`},
		{"object with content that is not a string", `{"content": 19}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := tinyWithConfig(t, `{"eos_token": `+tt.eosToken+`}`)
			_, err := Load(dir)
			if err == nil {
				t.Fatal("Load succeeded, want an error")
			}
			for _, want := range []string{"tokenizer_config.json", "eos_token"} {
				if !strings.Contains(err.Error(), want) {
					t.Errorf("Load error = %q, want it to mention %s", err, want)
				}
			}
		})
	}
}

func TestLoadRejectsMalformedConfig(t *testing.T) {
	dir := tinyWithConfig(t, `{"eos_token": `)
	_, err := Load(dir)
	if err == nil {
		t.Fatal("Load succeeded, want an error")
	}
	if !strings.Contains(err.Error(), "tokenizer_config.json") {
		t.Errorf("Load error = %q, want it to mention tokenizer_config.json", err)
	}
}

// Load must refuse a pipeline this package does not implement, rather than
// load it and tokenize differently from the reference.
func TestLoadRejectsUnsupportedSettings(t *testing.T) {
	tests := []struct {
		path  string // setting to change in the tiny tokenizer.json
		value any
		want  string // setting the error must name
	}{
		{"normalizer", nil, "normalizer.type"},
		{"normalizer.type", "NFKC", "normalizer.type"},
		{"pre_tokenizer", nil, "pre_tokenizer"},
		{"pre_tokenizer.type", "ByteLevel", "pre_tokenizer"},
		{"pre_tokenizer.pretokenizers", []any{}, "pre_tokenizer"},
		{"pre_tokenizer.pretokenizers.0.type", "Punctuation", "pre_tokenizer.pretokenizers[0].type"},
		{"pre_tokenizer.pretokenizers.0.pattern", map[string]any{"String": " "}, "pre_tokenizer.pretokenizers[0].pattern"},
		{"pre_tokenizer.pretokenizers.0.behavior", "Removed", "pre_tokenizer.pretokenizers[0].behavior"},
		{"pre_tokenizer.pretokenizers.0.invert", true, "pre_tokenizer.pretokenizers[0].invert"},
		{"pre_tokenizer.pretokenizers.1.type", "Whitespace", "pre_tokenizer.pretokenizers[1].type"},
		{"pre_tokenizer.pretokenizers.1.add_prefix_space", true, "pre_tokenizer.pretokenizers[1].add_prefix_space"},
		{"pre_tokenizer.pretokenizers.1.use_regex", true, "pre_tokenizer.pretokenizers[1].use_regex"},
		{"pre_tokenizer.pretokenizers.1.use_regex", absent{}, "pre_tokenizer.pretokenizers[1].use_regex"},
		{"post_processor.type", "TemplateProcessing", "post_processor.type"},
		{"decoder", nil, "decoder.type"},
		{"decoder.type", "Metaspace", "decoder.type"},
		{"model.type", "WordPiece", "model.type"},
		{"model.dropout", 0.1, "model.dropout"},
		{"model.unk_token", "<unk>", "model.unk_token"},
		{"model.continuing_subword_prefix", "##", "model.continuing_subword_prefix"},
		{"model.end_of_word_suffix", "</w>", "model.end_of_word_suffix"},
		{"model.byte_fallback", true, "model.byte_fallback"},
		{"model.ignore_merges", true, "model.ignore_merges"},
		{"added_tokens.0.single_word", true, "added_tokens[0].single_word"},
		{"added_tokens.1.lstrip", true, "added_tokens[1].lstrip"},
		{"added_tokens.1.rstrip", true, "added_tokens[1].rstrip"},
		{"added_tokens.2.normalized", true, "added_tokens[2].normalized"},
		{"added_tokens.3.content", "", "added_tokens[3].content"},
	}
	for _, tt := range tests {
		t.Run(fmt.Sprintf("%s=%v", tt.path, tt.value), func(t *testing.T) {
			dir := tinyWithSetting(t, tt.path, tt.value)
			_, err := Load(dir)
			if err == nil {
				t.Fatal("Load succeeded, want an error")
			}
			if !strings.Contains(err.Error(), "unsupported "+tt.want) {
				t.Errorf("Load error = %q, want it to say: unsupported %s", err, tt.want)
			}
		})
	}
}

func TestLoadAcceptsAbsentPostProcessor(t *testing.T) {
	dir := tinyWithSetting(t, "post_processor", nil)
	if _, err := Load(dir); err != nil {
		t.Errorf("Load error: %v", err)
	}
}

func TestLoadRejectsInvalidSplitPattern(t *testing.T) {
	dir := tinyWithSetting(t, "pre_tokenizer.pretokenizers.0.pattern.Regex", "(")
	_, err := Load(dir)
	if err == nil {
		t.Fatal("Load succeeded, want an error")
	}
	if !strings.Contains(err.Error(), "split pattern") {
		t.Errorf("Load error = %q, want it to mention the split pattern", err)
	}
}

// Older tokenizer.json files, Qwen2.5's among them, write each merge as one
// "left right" string instead of a ["left", "right"] pair.
func TestLoadReadsMergesWrittenAsStrings(t *testing.T) {
	dir := tinyWithSetting(t, "model.merges", []any{"b c", "a b", "a bc"})
	tok, err := Load(dir)
	if err != nil {
		t.Fatalf("Load error: %v", err)
	}
	if got, want := tok.Encode("abc"), []int32{17}; !slices.Equal(got, want) {
		t.Errorf("Encode(%q) = %v, want %v", "abc", got, want)
	}
}

func TestLoadRejectsBadMerges(t *testing.T) {
	tests := []struct {
		name   string
		merges any
		want   string // what the error must mention
	}{
		{"left token outside vocabulary", []any{[]any{"x", "c"}}, `"x"`},
		{"right token outside vocabulary", []any{[]any{"a", "x"}}, `"x"`},
		{"joined token outside vocabulary", []any{[]any{"a", "c"}}, `"ac"`},
		{"string with three parts", []any{"a b c"}, "merges"},
		{"string with one part", []any{"ab"}, "merges"},
		{"pair with three parts", []any{[]any{"a", "b", "c"}}, "merges"},
		{"pair with one part", []any{[]any{"ab"}}, "merges"},
		{"not a list of merges", []any{1, 2}, "merges"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := tinyWithSetting(t, "model.merges", tt.merges)
			_, err := Load(dir)
			if err == nil {
				t.Fatal("Load succeeded, want an error")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("Load error = %q, want it to mention %s", err, tt.want)
			}
		})
	}
}

// The tiny tokenizer has 23 tokens, so its ids run from 0 to 22.
func TestLoadRejectsTokenIDsOutOfRange(t *testing.T) {
	tests := []struct {
		path string // id to change in the tiny tokenizer.json
		id   int
		want string // token the error must name
	}{
		{"model.vocab.a", -1, `"a"`},
		{"model.vocab.a", 23, `"a"`},
		{"added_tokens.1.id", -1, `"<|im_end|>"`},
		{"added_tokens.1.id", 2_000_000_000, `"<|im_end|>"`},
	}
	for _, tt := range tests {
		t.Run(fmt.Sprintf("%s=%d", tt.path, tt.id), func(t *testing.T) {
			dir := tinyWithSetting(t, tt.path, tt.id)
			_, err := Load(dir)
			if err == nil {
				t.Fatal("Load succeeded, want an error")
			}
			if !strings.Contains(err.Error(), tt.want) || !strings.Contains(err.Error(), fmt.Sprint(tt.id)) {
				t.Errorf("Load error = %q, want it to name token %s and id %d", err, tt.want, tt.id)
			}
		})
	}
}

func TestLoadRejectsMissingVocabulary(t *testing.T) {
	dir := tinyWithSetting(t, "model.vocab", absent{})
	_, err := Load(dir)
	if err == nil {
		t.Fatal("Load succeeded, want an error")
	}
	if !strings.Contains(err.Error(), "model.vocab") {
		t.Errorf("Load error = %q, want it to mention model.vocab", err)
	}
}

// A vocabulary may hold an empty token. With no eos_token configured, that
// token must not be mistaken for it.
func TestEOSTokenIsNotTheEmptyToken(t *testing.T) {
	dir := tinyWithSetting(t, "model.vocab.", 23)
	tok, err := Load(dir)
	if err != nil {
		t.Fatalf("Load error: %v", err)
	}
	if token, id, ok := tok.EOSToken(); ok {
		t.Errorf("EOSToken() = %q, %d, true, want not ok", token, id)
	}
}
