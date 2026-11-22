package sample

import (
	"encoding/json"
	"math"
	"os"
	"slices"
	"strconv"
	"testing"

	"github.com/bigshort2022/mlx-lm-go/internal/models/modeltest"
	"github.com/bigshort2022/mlx-lm-go/mlx"
)

// goldenCase is one entry of testdata/sampler/cases.json: a filter or logits
// processor of mlx_lm.sample_utils with its parameters. The expected output
// is the array of the same name in cases.safetensors.
type goldenCase struct {
	Name        string             `json:"name"`
	Kind        string             `json:"kind"`
	TopK        int                `json:"top_k"`
	TopP        float32            `json:"top_p"`
	MinP        float32            `json:"min_p"`
	Penalty     float32            `json:"penalty"`
	ContextSize int                `json:"context_size"`
	LogitBias   map[string]float32 `json:"logit_bias"`
}

// apply runs the case on the inputs of cases.safetensors.
func (c goldenCase) apply(t *testing.T, in map[string]*mlx.Array) *mlx.Array {
	t.Helper()
	logits, logprobs, history := in["logits"], in["logprobs"], in["history"]
	switch c.Kind {
	case "top_k":
		out, err := TopK(logprobs, c.TopK)
		if err != nil {
			t.Fatal(err)
		}
		return out
	case "top_p":
		return TopP(logprobs, c.TopP)
	case "min_p":
		return MinP(logprobs, c.MinP)
	case "repetition_penalty":
		return RepetitionPenalty(history, logits, c.Penalty, c.ContextSize)
	case "presence_penalty":
		return PresencePenalty(history, logits, c.Penalty, c.ContextSize)
	case "frequency_penalty":
		return FrequencyPenalty(history, logits, c.Penalty, c.ContextSize)
	case "logit_bias":
		bias := map[int32]float32{}
		for id, v := range c.LogitBias {
			n, err := strconv.Atoi(id)
			if err != nil {
				t.Fatal(err)
			}
			bias[int32(n)] = v
		}
		out, err := LogitBias(logits, bias)
		if err != nil {
			t.Fatal(err)
		}
		return out
	default:
		t.Fatalf("unknown kind %q", c.Kind)
		return nil
	}
}

// inScope pins the test and runs fn inside a scope that must not fail.
func inScope(t *testing.T, fn func()) {
	t.Helper()
	modeltest.Pin(t)
	err := mlx.Scoped(func() error {
		fn()
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

// wantFiltered fails the test unless got has -inf exactly where want has it
// and is within tol of want elsewhere.
func wantFiltered(t *testing.T, got *mlx.Array, want []float32, tol float64) {
	t.Helper()
	if shape := got.Shape(); !slices.Equal(shape, []int{1, len(want)}) {
		t.Fatalf("result has shape %v, want [1 %d]", shape, len(want))
	}
	values := mlx.AsType(got, mlx.Float32).Float32s()
	for i, w := range want {
		g := values[i]
		if gInf, wInf := math.IsInf(float64(g), -1), math.IsInf(float64(w), -1); gInf || wInf {
			if gInf != wInf {
				t.Errorf("element %d is %v, want %v", i, g, w)
			}
			continue
		}
		if diff := math.Abs(float64(g) - float64(w)); !(diff <= tol) {
			t.Errorf("element %d is %v, want %v (differs by %g)", i, g, w, diff)
		}
	}
}

func TestGoldenCases(t *testing.T) {
	data, err := os.ReadFile(modeltest.Testdata("sampler", "cases.json"))
	if err != nil {
		t.Fatal(err)
	}
	var file struct {
		Cases []goldenCase `json:"cases"`
	}
	if err := json.Unmarshal(data, &file); err != nil {
		t.Fatal(err)
	}
	if len(file.Cases) == 0 {
		t.Fatal("cases.json has no cases")
	}
	for _, c := range file.Cases {
		t.Run(c.Name, func(t *testing.T) {
			inScope(t, func() {
				arrays, _, err := mlx.LoadSafetensors(modeltest.Testdata("sampler", "cases.safetensors"))
				if err != nil {
					t.Fatal(err)
				}
				want, ok := arrays[c.Name]
				if !ok {
					t.Fatalf("cases.safetensors has no array %q", c.Name)
				}
				wantFiltered(t, c.apply(t, arrays), want.Float32s(), 1e-4)
				for _, a := range arrays {
					a.Free()
				}
			})
		})
	}
}
