package tokenizer

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// tokenizerFile is the part of tokenizer.json this package reads.
type tokenizerFile struct {
	AddedTokens  []addedTokenFile `json:"added_tokens"`
	Normalizer   componentFile    `json:"normalizer"`
	PreTokenizer struct {
		Type          string             `json:"type"`
		PreTokenizers []preTokenizerFile `json:"pretokenizers"`
	} `json:"pre_tokenizer"`
	PostProcessor componentFile `json:"post_processor"`
	Decoder       componentFile `json:"decoder"`
	Model         modelFile     `json:"model"`
}

// componentFile is a pipeline stage of which only the kind matters here. A
// stage that is null in the file has an empty Type.
type componentFile struct {
	Type string `json:"type"`
}

// preTokenizerFile is one entry of a "Sequence" pre-tokenizer: either a
// "Split" or a "ByteLevel", each with its own settings.
type preTokenizerFile struct {
	Type string `json:"type"`

	// Split settings.
	Pattern struct {
		Regex string `json:"Regex"`
	} `json:"pattern"`
	Behavior string `json:"behavior"`
	Invert   bool   `json:"invert"`

	// ByteLevel settings. The reference treats an absent use_regex as true.
	AddPrefixSpace bool  `json:"add_prefix_space"`
	UseRegex       *bool `json:"use_regex"`
}

// modelFile is tokenizer.json "model".
type modelFile struct {
	Type                    string           `json:"type"`
	Dropout                 *float64         `json:"dropout"`
	UnkToken                *string          `json:"unk_token"`
	ContinuingSubwordPrefix string           `json:"continuing_subword_prefix"`
	EndOfWordSuffix         string           `json:"end_of_word_suffix"`
	ByteFallback            bool             `json:"byte_fallback"`
	IgnoreMerges            bool             `json:"ignore_merges"`
	Vocab                   map[string]int32 `json:"vocab"`
	Merges                  mergeList        `json:"merges"`
}

// mergeList is tokenizer.json "merges": the left and right token of each
// merge, in rank order. Newer files write a merge as ["left", "right"], older
// ones as the string "left right".
type mergeList [][2]string

// UnmarshalJSON accepts either spelling of the merge list.
func (m *mergeList) UnmarshalJSON(data []byte) error {
	var merges [][]string
	if err := json.Unmarshal(data, &merges); err != nil {
		var lines []string
		if err := json.Unmarshal(data, &lines); err != nil {
			return errors.New(`merges must be ["left", "right"] pairs or "left right" strings`)
		}
		merges = make([][]string, len(lines))
		for i, line := range lines {
			merges[i] = strings.Split(line, " ")
		}
	}

	list := make(mergeList, len(merges))
	for i, parts := range merges {
		if len(parts) != 2 {
			return fmt.Errorf("merges[%d] names %d tokens, want 2", i, len(parts))
		}
		list[i] = [2]string(parts)
	}
	*m = list
	return nil
}

// addedTokenFile is one entry of tokenizer.json "added_tokens".
type addedTokenFile struct {
	ID         int32  `json:"id"`
	Content    string `json:"content"`
	SingleWord bool   `json:"single_word"`
	LStrip     bool   `json:"lstrip"`
	RStrip     bool   `json:"rstrip"`
	Normalized bool   `json:"normalized"`
}

// configFile is the part of tokenizer_config.json this package reads.
type configFile struct {
	// EOSToken is eos_token as written, nil when the file has none.
	EOSToken json.RawMessage `json:"eos_token"`
}

// eosToken returns the token that eos_token names. The file writes it either
// as a string or as a serialized AddedToken, an object with the token in
// "content". ok is false when eos_token is absent or null.
func (c *configFile) eosToken() (token string, ok bool, err error) {
	if c.EOSToken == nil {
		return "", false, nil
	}
	var name *string
	if json.Unmarshal(c.EOSToken, &name) == nil {
		if name == nil {
			return "", false, nil
		}
		return *name, true, nil
	}
	var added struct {
		Content *string `json:"content"`
	}
	if json.Unmarshal(c.EOSToken, &added) != nil || added.Content == nil {
		return "", false, errors.New(`eos_token must be a string or an object with a string "content"`)
	}
	return *added.Content, true, nil
}

// Load reads tokenizer.json and tokenizer_config.json from dir. A directory
// without tokenizer_config.json loads as a tokenizer with no EOS token.
//
// Load implements one pipeline, the one the Qwen tokenizers use: NFC
// normalisation, a regex Split followed by ByteLevel, plain BPE, and ByteLevel
// decoding. It fails on a tokenizer.json that asks for anything else.
func Load(dir string) (*Tokenizer, error) {
	path := filepath.Join(dir, "tokenizer.json")
	var file tokenizerFile
	if err := readJSON(path, &file); err != nil {
		return nil, err
	}
	tok, err := newTokenizer(&file)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}

	configPath := filepath.Join(dir, "tokenizer_config.json")
	var config configFile
	if err := readJSON(configPath, &config); err != nil && !errors.Is(err, fs.ErrNotExist) {
		return nil, err
	}
	eos, ok, err := config.eosToken()
	if err != nil {
		return nil, fmt.Errorf("%s: %w", configPath, err)
	}
	if ok {
		id, found := tok.ids[eos]
		if !found {
			return nil, fmt.Errorf("%s: eos_token %q is not in the vocabulary", configPath, eos)
		}
		tok.eos, tok.eosID, tok.hasEOS = eos, id, true
	}
	return tok, nil
}

// newTokenizer builds a tokenizer, as yet without an EOS token, from a parsed
// tokenizer.json.
func newTokenizer(file *tokenizerFile) (*Tokenizer, error) {
	if setting := file.unsupportedSetting(); setting != "" {
		return nil, fmt.Errorf("unsupported %s", setting)
	}
	vocab := file.Model.Vocab
	if vocab == nil {
		return nil, errors.New("model.vocab is missing")
	}
	splitter, err := newSplitter(file.PreTokenizer.PreTokenizers[0].Pattern.Regex)
	if err != nil {
		return nil, err
	}
	model, err := newBPE(vocab, file.Model.Merges)
	if err != nil {
		return nil, err
	}
	decodeTable, err := newDecodeTable(vocab, file.AddedTokens)
	if err != nil {
		return nil, err
	}

	// Everything that needed the base vocabulary on its own is built, so the
	// same map can go on to hold the added tokens too.
	ids := vocab
	for _, added := range file.AddedTokens {
		ids[added.Content] = added.ID
	}
	return &Tokenizer{
		ids:         ids,
		added:       newAddedTokens(file.AddedTokens),
		splitter:    splitter,
		bpe:         model,
		decodeTable: decodeTable,
	}, nil
}

// unsupportedSetting names the first setting of the file that asks for
// behaviour this package does not implement, or returns "" when there is none.
// Loading such a file anyway would tokenize differently from the reference
// without any sign of it.
func (f *tokenizerFile) unsupportedSetting() string {
	if f.PreTokenizer.Type != "Sequence" || len(f.PreTokenizer.PreTokenizers) != 2 {
		return "pre_tokenizer (want a Sequence of Split and ByteLevel)"
	}
	split, byteLevel := f.PreTokenizer.PreTokenizers[0], f.PreTokenizer.PreTokenizers[1]
	postProcessor := f.PostProcessor.Type

	type setting struct {
		name      string
		supported bool
	}
	settings := []setting{
		{"normalizer.type (want NFC)", f.Normalizer.Type == "NFC"},
		{"pre_tokenizer.pretokenizers[0].type (want Split)", split.Type == "Split"},
		{"pre_tokenizer.pretokenizers[0].pattern (want a Regex)", split.Pattern.Regex != ""},
		{"pre_tokenizer.pretokenizers[0].behavior (want Isolated)", split.Behavior == "Isolated"},
		{"pre_tokenizer.pretokenizers[0].invert (want false)", !split.Invert},
		{"pre_tokenizer.pretokenizers[1].type (want ByteLevel)", byteLevel.Type == "ByteLevel"},
		{"pre_tokenizer.pretokenizers[1].add_prefix_space (want false)", !byteLevel.AddPrefixSpace},
		{"pre_tokenizer.pretokenizers[1].use_regex (want false)", byteLevel.UseRegex != nil && !*byteLevel.UseRegex},
		{"post_processor.type (want ByteLevel or none)", postProcessor == "" || postProcessor == "ByteLevel"},
		{"decoder.type (want ByteLevel)", f.Decoder.Type == "ByteLevel"},
		{"model.type (want BPE)", f.Model.Type == "BPE"},
		{"model.dropout (want null)", f.Model.Dropout == nil},
		{"model.unk_token (want null)", f.Model.UnkToken == nil},
		{"model.continuing_subword_prefix (want none)", f.Model.ContinuingSubwordPrefix == ""},
		{"model.end_of_word_suffix (want none)", f.Model.EndOfWordSuffix == ""},
		{"model.byte_fallback (want false)", !f.Model.ByteFallback},
		{"model.ignore_merges (want false)", !f.Model.IgnoreMerges},
	}
	for i, added := range f.AddedTokens {
		settings = append(settings,
			setting{fmt.Sprintf("added_tokens[%d].content (want text)", i), added.Content != ""},
			setting{fmt.Sprintf("added_tokens[%d].single_word (want false)", i), !added.SingleWord},
			setting{fmt.Sprintf("added_tokens[%d].lstrip (want false)", i), !added.LStrip},
			setting{fmt.Sprintf("added_tokens[%d].rstrip (want false)", i), !added.RStrip},
			setting{fmt.Sprintf("added_tokens[%d].normalized (want false)", i), !added.Normalized},
		)
	}
	for _, s := range settings {
		if !s.supported {
			return s.name
		}
	}
	return ""
}

// readJSON decodes the JSON file at path into v.
func readJSON(path string, v any) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(data, v); err != nil {
		return fmt.Errorf("parsing %s: %w", path, err)
	}
	return nil
}
