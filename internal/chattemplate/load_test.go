package chattemplate

import (
	"maps"
	"os"
	"path/filepath"
	"testing"

	"github.com/bigshort2022/mlx-lm-go/internal/hub"
)

// modelDir writes the given files into a temporary model directory.
func modelDir(t *testing.T, files map[string]string) string {
	t.Helper()
	dir := t.TempDir()
	for name, content := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func renderLoaded(t *testing.T, dir string) string {
	t.Helper()
	tmpl, err := Load(dir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	got, err := tmpl.Render(Input{Messages: oneUserMessage})
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	return got
}

func TestLoadReadsTemplateFromTokenizerConfig(t *testing.T) {
	dir := modelDir(t, map[string]string{
		"tokenizer_config.json": `{"chat_template": "from config"}`,
	})
	if got, want := renderLoaded(t, dir), "from config"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestLoadPrefersChatTemplateJinja(t *testing.T) {
	dir := modelDir(t, map[string]string{
		"tokenizer_config.json": `{"chat_template": "from config"}`,
		"chat_template.jinja":   "from jinja file",
	})
	if got, want := renderLoaded(t, dir), "from jinja file"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestLoadExposesSpecialTokens(t *testing.T) {
	dir := modelDir(t, map[string]string{
		"tokenizer_config.json": `{
			"bos_token": {"content": "<s>", "lstrip": false},
			"eos_token": "</s>",
			"pad_token": null,
			"chat_template": "{{ bos_token }}|{{ eos_token }}|{{ pad_token is defined }}|{{ unk_token is defined }}"
		}`,
	})
	if got, want := renderLoaded(t, dir), "<s>|</s>|False|False"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestLoadFailsWithoutAnyTemplate(t *testing.T) {
	dir := modelDir(t, map[string]string{
		"tokenizer_config.json": `{"eos_token": "</s>"}`,
	})
	if _, err := Load(dir); err == nil {
		t.Fatal("Load succeeded for a model with no chat template")
	}
}

// The copies under testdata are what every fixture test renders with. This
// checks them against the models in the local Hugging Face cache, and is the
// only test that needs the cache.
func TestVendoredTemplatesMatchTheModels(t *testing.T) {
	for _, family := range goldenFamilies {
		t.Run(family.name, func(t *testing.T) {
			dir, err := hub.Resolve(family.repo)
			if err != nil {
				t.Skipf("model not available: %v", err)
			}
			wantSource, wantTokens, err := readModel(dir)
			if err != nil {
				t.Fatalf("readModel(%s): %v", dir, err)
			}
			gotSource, gotTokens, err := readModel(vendoredDir(family.name))
			if err != nil {
				t.Fatalf("readModel(%s): %v", vendoredDir(family.name), err)
			}
			if gotSource != wantSource {
				t.Errorf("vendored template differs from the model's: %s", firstDifference(gotSource, wantSource))
			}
			if !maps.Equal(gotTokens, wantTokens) {
				t.Errorf("vendored special tokens = %v, the model's are %v", gotTokens, wantTokens)
			}
		})
	}
}

func TestLoadSourceReadsSpecialTokensFromTheModel(t *testing.T) {
	tmpl, err := LoadSource(vendoredDir("qwen2"), "{{ eos_token }}|{{ messages[0].content }}")
	if err != nil {
		t.Fatal(err)
	}
	got, err := tmpl.Render(Input{Messages: oneUserMessage})
	if err != nil {
		t.Fatal(err)
	}
	if want := "<|im_end|>|hi"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

// As with Load, a model without tokenizer_config.json has no special tokens.
func TestLoadSourceWithoutTokenizerConfig(t *testing.T) {
	tmpl, err := LoadSource(t.TempDir(), "{{ eos_token is defined }}")
	if err != nil {
		t.Fatal(err)
	}
	got, err := tmpl.Render(Input{Messages: oneUserMessage})
	if err != nil {
		t.Fatal(err)
	}
	if want := "False"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestLoadSourceRejectsMalformedSource(t *testing.T) {
	if _, err := LoadSource(vendoredDir("qwen2"), "{% if %}"); err == nil {
		t.Fatal("LoadSource accepted a malformed template")
	}
}

func TestLoadSourceRejectsMalformedTokenizerConfig(t *testing.T) {
	dir := modelDir(t, map[string]string{"tokenizer_config.json": `{"eos_token": `})
	if _, err := LoadSource(dir, "static"); err == nil {
		t.Fatal("LoadSource accepted a malformed tokenizer_config.json")
	}
}
