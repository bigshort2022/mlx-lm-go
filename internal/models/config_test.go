package models

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/bigshort2022/mlx-lm-go/internal/models/modeltest"
)

// configDir returns a temporary model directory holding the given files.
func configDir(t *testing.T, files map[string]string) string {
	t.Helper()
	dir := t.TempDir()
	for name, content := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func TestLoadConfigReadsQuantizationAndEOS(t *testing.T) {
	dir := modeltest.Dir(t)
	cfg, err := LoadConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.ModelType != "qwen2" {
		t.Errorf("ModelType = %q, want qwen2", cfg.ModelType)
	}
	want := Quantization{GroupSize: 64, Bits: 4, Mode: "affine"}
	if cfg.Quantization == nil || *cfg.Quantization != want {
		t.Errorf("Quantization = %+v, want %+v", cfg.Quantization, want)
	}
	if !slices.Equal(cfg.EOSTokenIDs, []int32{151645}) {
		t.Errorf("EOSTokenIDs = %v, want [151645]", cfg.EOSTokenIDs)
	}
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	if string(cfg.Raw) != string(raw) {
		t.Errorf("Raw is not the content of config.json")
	}
}

func TestConfigEOSAcceptsNumberOrList(t *testing.T) {
	tests := []struct {
		name, config string
		want         []int32
	}{
		{"number", `{"model_type":"m","eos_token_id":7}`, []int32{7}},
		{"list", `{"model_type":"m","eos_token_id":[7,9]}`, []int32{7, 9}},
		{"absent", `{"model_type":"m"}`, nil},
		{"null", `{"model_type":"m","eos_token_id":null}`, nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg, err := LoadConfig(configDir(t, map[string]string{"config.json": tt.config}))
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(cfg.EOSTokenIDs, tt.want) {
				t.Errorf("EOSTokenIDs = %v, want %v", cfg.EOSTokenIDs, tt.want)
			}
		})
	}
}

func TestConfigEOSRejectsOtherTypes(t *testing.T) {
	_, err := LoadConfig(configDir(t, map[string]string{"config.json": `{"model_type":"m","eos_token_id":"seven"}`}))
	if err == nil {
		t.Fatal("LoadConfig accepted a string eos_token_id")
	}
}

func TestConfigQuantizationAbsent(t *testing.T) {
	cfg, err := LoadConfig(configDir(t, map[string]string{"config.json": `{"model_type":"m"}`}))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Quantization != nil {
		t.Errorf("Quantization = %+v, want nil", cfg.Quantization)
	}
}

func TestConfigQuantizationKeepsItsMode(t *testing.T) {
	cfg, err := LoadConfig(configDir(t, map[string]string{
		"config.json": `{"model_type":"m","quantization":{"group_size":32,"bits":8,"mode":"mxfp8"}}`,
	}))
	if err != nil {
		t.Fatal(err)
	}
	want := Quantization{GroupSize: 32, Bits: 8, Mode: "mxfp8"}
	if cfg.Quantization == nil || *cfg.Quantization != want {
		t.Errorf("Quantization = %+v, want %+v", cfg.Quantization, want)
	}
}

// mlx_lm.utils.load_config lets generation_config.json override the
// end-of-sequence tokens of config.json.
func TestConfigEOSFromGenerationConfigWins(t *testing.T) {
	cfg, err := LoadConfig(configDir(t, map[string]string{
		"config.json":            `{"model_type":"m","eos_token_id":7}`,
		"generation_config.json": `{"eos_token_id":[7,9]}`,
	}))
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(cfg.EOSTokenIDs, []int32{7, 9}) {
		t.Errorf("EOSTokenIDs = %v, want [7 9]", cfg.EOSTokenIDs)
	}
}

func TestConfigIgnoresBrokenGenerationConfig(t *testing.T) {
	cfg, err := LoadConfig(configDir(t, map[string]string{
		"config.json":            `{"model_type":"m","eos_token_id":7}`,
		"generation_config.json": `{not json`,
	}))
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(cfg.EOSTokenIDs, []int32{7}) {
		t.Errorf("EOSTokenIDs = %v, want [7]", cfg.EOSTokenIDs)
	}
}

func TestLoadConfigReportsMissingFile(t *testing.T) {
	if _, err := LoadConfig(t.TempDir()); err == nil {
		t.Fatal("LoadConfig succeeded without a config.json")
	}
}
