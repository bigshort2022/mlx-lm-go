package tokenizer

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestQwen38PromptIDsMatchPython(t *testing.T) {
	fixture := filepath.Join("..", "chattemplate", "testdata", "qwen38")
	var meta struct {
		Repo     string `json:"repo"`
		Snapshot string `json:"snapshot"`
	}
	data, err := os.ReadFile(filepath.Join(fixture, "meta.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &meta); err != nil {
		t.Fatal(err)
	}
	if meta.Repo == "" || meta.Snapshot == "" {
		t.Fatal("missing repo or snapshot in Qwen3.8 fixture metadata")
	}
	cache := os.Getenv("HF_HUB_CACHE")
	if cache == "" {
		home := os.Getenv("HF_HOME")
		if home == "" {
			userHome, err := os.UserHomeDir()
			if err != nil {
				t.Fatal(err)
			}
			home = filepath.Join(userHome, ".cache", "huggingface")
		}
		cache = filepath.Join(home, "hub")
	}
	dir := filepath.Join(cache, "models--"+strings.ReplaceAll(meta.Repo, "/", "--"), "snapshots", meta.Snapshot)
	if _, err := os.Stat(dir); errors.Is(err, os.ErrNotExist) {
		t.Skipf("pinned tokenizer snapshot not cached: %s", dir)
	} else if err != nil {
		t.Fatal(err)
	}
	tok, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}

	rows, err := os.Open(filepath.Join(fixture, "template.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	successes := 0
	for dec := json.NewDecoder(rows); dec.More(); {
		var row struct {
			Name   string  `json:"name"`
			Prompt *string `json:"prompt"`
			IDs    []int32 `json:"prompt_ids"`
			Error  string  `json:"error"`
		}
		if err := dec.Decode(&row); err != nil {
			t.Fatal(err)
		}
		if row.Error != "" && row.Prompt == nil && len(row.IDs) == 0 {
			continue // the Python template rejected the input before tokenization
		}
		successes++
		t.Run(row.Name, func(t *testing.T) {
			if row.Prompt == nil || len(row.IDs) == 0 || row.Error != "" {
				t.Fatalf("incomplete Python prompt outcome: %+v", row)
			}
			if got := tok.Encode(*row.Prompt); !slices.Equal(got, row.IDs) {
				t.Fatalf("Encode = %v, want %v", got, row.IDs)
			}
		})
	}
	if successes != 9 {
		t.Fatalf("got %d successful Python prompts, want 9", successes)
	}
}
