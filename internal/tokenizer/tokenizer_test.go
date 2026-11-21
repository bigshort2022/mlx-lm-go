package tokenizer

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/bigshort2022/mlx-lm-go/internal/hub"
)

// tinyDir holds a hand-made tokenizer with 18 base tokens and 5 added tokens,
// so tests that use it run without the Hugging Face cache.
const tinyDir = "testdata/tiny"

func loadTiny(t *testing.T) *Tokenizer {
	t.Helper()
	tok, err := Load(tinyDir)
	if err != nil {
		t.Fatalf("Load(%q) error: %v", tinyDir, err)
	}
	return tok
}

// families names the golden fixture directories under the repository's
// testdata/. Each was produced by the Python reference from a real tokenizer.
var families = []string{"qwen2", "qwen3_5"}

// goldenMeta is the part of a family's meta.json the tests use.
type goldenMeta struct {
	Repo          string `json:"repo"`
	VocabSize     int    `json:"vocab_size"`
	SpecialTokens struct {
		EOSToken   string `json:"eos_token"`
		EOSTokenID int32  `json:"eos_token_id"`
	} `json:"special_tokens"`
}

func goldenPath(family, name string) string {
	return filepath.Join("..", "..", "testdata", family, name)
}

func readGoldenMeta(t *testing.T, family string) goldenMeta {
	t.Helper()
	data, err := os.ReadFile(goldenPath(family, "meta.json"))
	if err != nil {
		t.Fatal(err)
	}
	var meta goldenMeta
	if err := json.Unmarshal(data, &meta); err != nil {
		t.Fatalf("parsing %s meta.json: %v", family, err)
	}
	return meta
}

// readGolden decodes every row of a family's JSON Lines fixture.
func readGolden[Row any](t *testing.T, family, name string) []Row {
	t.Helper()
	f, err := os.Open(goldenPath(family, name))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	var rows []Row
	for dec := json.NewDecoder(f); dec.More(); {
		var row Row
		if err := dec.Decode(&row); err != nil {
			t.Fatalf("%s/%s row %d: %v", family, name, len(rows), err)
		}
		rows = append(rows, row)
	}
	if len(rows) == 0 {
		t.Fatalf("%s/%s has no rows", family, name)
	}
	return rows
}

var (
	familyMu         sync.Mutex
	familyTokenizers = map[string]*Tokenizer{}
)

// loadFamily loads the real tokenizer a family's fixtures were made from. It
// skips the test when the model is not in the Hugging Face cache.
func loadFamily(t *testing.T, family string) *Tokenizer {
	t.Helper()
	familyMu.Lock()
	defer familyMu.Unlock()
	if tok, ok := familyTokenizers[family]; ok {
		return tok
	}

	repo := readGoldenMeta(t, family).Repo
	dir, err := hub.Resolve(repo)
	if err != nil {
		t.Skipf("%s tokenizer is not available: %v", family, err)
	}
	tok, err := Load(dir)
	if err != nil {
		t.Fatalf("Load(%q) error: %v", dir, err)
	}
	familyTokenizers[family] = tok
	return tok
}

// tinyWithConfig returns a temporary model directory holding the tiny
// tokenizer.json and a tokenizer_config.json with the given content.
func tinyWithConfig(t *testing.T, config string) string {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(tinyDir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	for name, content := range map[string][]byte{"tokenizer.json": data, "tokenizer_config.json": []byte(config)} {
		if err := os.WriteFile(filepath.Join(dir, name), content, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

// absent, as a value for tinyWithSetting, removes the setting from the file.
type absent struct{}

// tinyWithSetting returns a temporary model directory holding the tiny
// tokenizer.json with one setting replaced. path is the setting's location in
// the file, such as "pre_tokenizer.pretokenizers.1.use_regex".
func tinyWithSetting(t *testing.T, path string, value any) string {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(tinyDir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	var root any
	if err := json.Unmarshal(data, &root); err != nil {
		t.Fatal(err)
	}

	steps := strings.Split(path, ".")
	parent := root
	for _, step := range steps[:len(steps)-1] {
		switch node := parent.(type) {
		case map[string]any:
			parent = node[step]
		case []any:
			index, err := strconv.Atoi(step)
			if err != nil {
				t.Fatalf("path %q: %q is not an array index", path, step)
			}
			parent = node[index]
		default:
			t.Fatalf("path %q: nothing to descend into at %q", path, step)
		}
	}
	object, ok := parent.(map[string]any)
	if !ok {
		t.Fatalf("path %q does not end in an object field", path)
	}
	last := steps[len(steps)-1]
	if value == (absent{}) {
		delete(object, last)
	} else {
		object[last] = value
	}

	data, err = json.Marshal(root)
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), data, 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

// Run with -race. Encode, Decode, TokenID and NewStream may be called from
// many goroutines at once; each goroutine has its own Stream.
func TestTokenizerIsSafeForConcurrentUse(t *testing.T) {
	tok := loadTiny(t)

	const goroutines = 16
	failures := make([]string, goroutines)
	var wg sync.WaitGroup
	for g := range goroutines {
		wg.Go(func() {
			for range 200 {
				ids := tok.Encode("ab<|im_start|>abc e\u0301")
				if text := tok.Decode(ids); text != "ab<|im_start|>abc \u00e9" {
					failures[g] = "Decode(Encode(text)) = " + strconv.QuoteToASCII(text)
				}
				stream := tok.NewStream()
				var streamed strings.Builder
				for _, id := range ids {
					streamed.WriteString(stream.Add(id))
				}
				streamed.WriteString(stream.Flush())
				if streamed.String() != tok.Decode(ids) {
					failures[g] = "stream = " + strconv.QuoteToASCII(streamed.String())
				}
				if id, ok := tok.TokenID("abc"); !ok || id != 17 {
					failures[g] = "TokenID(abc) = " + strconv.Itoa(int(id))
				}
			}
		})
	}
	wg.Wait()

	for g, failure := range failures {
		if failure != "" {
			t.Errorf("goroutine %d: %s", g, failure)
		}
	}
}
