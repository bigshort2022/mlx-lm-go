// Package modeltest gives tests the reference model and the golden fixtures
// that Python mlx-lm produced from it (tools/oracle/make_model_fixtures.py).
package modeltest

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"testing"

	"github.com/bigshort2022/mlx-lm-go/internal/hub"
	"github.com/bigshort2022/mlx-lm-go/mlx"
)

// ModelID is the model the golden fixtures were made with.
const ModelID = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

// Pin locks the calling test to its OS thread until the test ends, as every
// test that calls package mlx outside an exec.Thread must. Subtests run on
// their own goroutine and pin themselves.
func Pin(t testing.TB) {
	t.Helper()
	runtime.LockOSThread()
	t.Cleanup(runtime.UnlockOSThread)
}

// Dir returns the directory of the reference model. It skips the test when
// the model is not on this machine.
func Dir(t testing.TB) string {
	t.Helper()
	dir, err := hub.Resolve(ModelID)
	if err != nil {
		t.Skip(err)
	}
	return dir
}

// Weights lazily loads the tensors of the reference model and frees them when
// the test ends. The test must be pinned.
func Weights(t testing.TB) map[string]*mlx.Array {
	t.Helper()
	return load(t, filepath.Join(Dir(t), "model.safetensors"))
}

// Layers loads testdata/qwen2/layers.safetensors, the activations of the
// reference model for one short input, and frees them when the test ends. The
// test must be pinned.
func Layers(t testing.TB) map[string]*mlx.Array {
	t.Helper()
	return load(t, Testdata("qwen2", "layers.safetensors"))
}

func load(t testing.TB, path string) map[string]*mlx.Array {
	t.Helper()
	arrays, _, err := mlx.LoadSafetensors(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		for _, a := range arrays {
			a.Free()
		}
	})
	return arrays
}

// Testdata returns the path of a file under the repository's testdata.
func Testdata(elem ...string) string {
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Join(filepath.Dir(file), "..", "..", "..", "testdata")
	return filepath.Join(append([]string{root}, elem...)...)
}

// Case is one prompt of testdata/qwen2/generation.json with what the
// reference model made of it.
type Case struct {
	PromptIDs   []int32   `json:"prompt_ids"`
	Top50IDs    []int32   `json:"top50_ids"`    // the 50 likeliest first tokens, best first
	Top50Logits []float32 `json:"top50_logits"` // their logits
	GreedyIDs   []int32   `json:"greedy_ids"`   // up to 64 tokens, ending with the end-of-sequence token if one came
	PeakMemory  uint64    `json:"peak_memory_bytes"`
}

// EndsWithEOS reports whether the greedy continuation reached an
// end-of-sequence token.
func (c Case) EndsWithEOS(eos []int32) bool {
	return len(c.GreedyIDs) > 0 && slices.Contains(eos, c.GreedyIDs[len(c.GreedyIDs)-1])
}

// Generation reads testdata/qwen2/generation.json and returns its cases and
// the end-of-sequence tokens of the reference model.
func Generation(t testing.TB) (cases []Case, eos []int32) {
	t.Helper()
	data, err := os.ReadFile(Testdata("qwen2", "generation.json"))
	if err != nil {
		t.Fatal(err)
	}
	var file struct {
		Meta struct {
			EOS []int32 `json:"eos_token_ids"`
		} `json:"meta"`
		Cases []Case `json:"cases"`
	}
	if err := json.Unmarshal(data, &file); err != nil {
		t.Fatal(err)
	}
	return file.Cases, file.Meta.EOS
}

// MaxAbsDiff returns the largest absolute difference between the elements of
// got and want, both converted to float32. When their shapes differ it reports
// a test error and returns +Inf. It never ends the test, so that it can be
// called on an exec.Thread.
func MaxAbsDiff(t testing.TB, got, want *mlx.Array) float64 {
	t.Helper()
	if !slices.Equal(got.Shape(), want.Shape()) {
		t.Errorf("shape is %v, want %v", got.Shape(), want.Shape())
		return math.Inf(1)
	}
	g, w := floats(got), floats(want)
	worst := 0.0
	for i := range g {
		diff := math.Abs(float64(g[i]) - float64(w[i]))
		if !(diff <= worst) { // also catches NaN
			worst = diff
		}
	}
	return worst
}

func floats(a *mlx.Array) []float32 {
	f := mlx.AsType(a, mlx.Float32)
	defer f.Free()
	return f.Float32s()
}
