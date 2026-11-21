package models

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
)

// Config is what the loader needs to know about a model, read from its
// config.json.
//
// EOSTokenIDs is the one field that another file can set: when
// generation_config.json exists and names end-of-sequence tokens, those
// replace the ones of config.json, as in mlx-lm's load_config.
type Config struct {
	ModelType    string
	EOSTokenIDs  []int32 // "eos_token_id", a number or a list, of generation_config.json or else config.json; empty when absent
	Quantization *Quantization
	Raw          json.RawMessage // the whole config.json, for the architecture package
}

// Quantization describes how the quantized weights of a model were made.
type Quantization struct {
	GroupSize int
	Bits      int
	Mode      string // "affine" when absent
}

// tokenIDs decodes a JSON value that is one token id, a list of them or null.
type tokenIDs []int32

func (ids *tokenIDs) UnmarshalJSON(data []byte) error {
	if string(data) == "null" {
		*ids = nil
		return nil
	}
	var one int32
	if err := json.Unmarshal(data, &one); err == nil {
		*ids = tokenIDs{one}
		return nil
	}
	var many []int32
	if err := json.Unmarshal(data, &many); err != nil {
		return fmt.Errorf("eos_token_id is %s, want a token id or a list of token ids", data)
	}
	*ids = many
	return nil
}

// LoadConfig reads the config.json of the model in dir. Like
// mlx_lm.utils.load_config, it prefers the end-of-sequence tokens of
// generation_config.json when that file names any.
func LoadConfig(dir string) (Config, error) {
	path := filepath.Join(dir, "config.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		return Config{}, err
	}
	var file struct {
		ModelType    string   `json:"model_type"`
		EOS          tokenIDs `json:"eos_token_id"`
		Quantization *struct {
			GroupSize int    `json:"group_size"`
			Bits      int    `json:"bits"`
			Mode      string `json:"mode"`
		} `json:"quantization"`
	}
	if err := json.Unmarshal(raw, &file); err != nil {
		return Config{}, fmt.Errorf("%s: %w", path, err)
	}

	cfg := Config{ModelType: file.ModelType, EOSTokenIDs: file.EOS, Raw: raw}
	if q := file.Quantization; q != nil {
		cfg.Quantization = &Quantization{GroupSize: q.GroupSize, Bits: q.Bits, Mode: q.Mode}
		if q.Mode == "" {
			cfg.Quantization.Mode = "affine"
		}
	}
	eos, err := generationEOS(dir)
	if err != nil {
		return Config{}, err
	}
	if len(eos) > 0 {
		cfg.EOSTokenIDs = eos
	}
	return cfg, nil
}

// generationEOS returns the end-of-sequence tokens of generation_config.json,
// or none when the file is absent or, as in the reference, unreadable as JSON.
func generationEOS(dir string) ([]int32, error) {
	data, err := os.ReadFile(filepath.Join(dir, "generation_config.json"))
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var file struct {
		EOS tokenIDs `json:"eos_token_id"`
	}
	if json.Unmarshal(data, &file) != nil {
		return nil, nil
	}
	return file.EOS, nil
}
