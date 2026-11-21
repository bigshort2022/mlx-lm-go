package models_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/bigshort2022/mlx-lm-go/internal/models"
	_ "github.com/bigshort2022/mlx-lm-go/internal/models/qwen2"
)

func TestLoadRejectsUnknownModelType(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"model_type":"nope"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	_, _, err := models.Load(dir)
	if err == nil {
		t.Fatal("Load succeeded for model_type nope")
	}
	for _, want := range []string{"nope", "qwen2"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not mention %q", err, want)
		}
	}
}
