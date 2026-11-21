package chattemplate

import (
	"os"
	"path/filepath"
	"testing"
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
