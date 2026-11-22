package models_test

import (
	"cmp"
	"context"
	"fmt"
	"math"
	"os"
	"slices"
	"sync"
	"testing"

	"github.com/bigshort2022/mlx-lm-go/internal/exec"
	"github.com/bigshort2022/mlx-lm-go/internal/models"
	"github.com/bigshort2022/mlx-lm-go/internal/models/modeltest"
	"github.com/bigshort2022/mlx-lm-go/mlx"
)

// The tests of the whole model share one inference thread and one copy of the
// reference model, loaded on that thread by the first test that needs it.
var (
	thread    *exec.Thread
	loadOnce  sync.Once
	reference models.Model
	loadErr   error
)

func TestMain(m *testing.M) {
	thread = exec.Start()
	code := m.Run()
	thread.Close()
	os.Exit(code)
}

// onModel runs fn on the inference thread, inside a scope, with the reference
// model. fn reports failures with t.Error or by returning an error; it must
// not end the test itself, which would end the thread.
func onModel(t *testing.T, fn func(model models.Model) error) {
	t.Helper()
	dir := modeltest.Dir(t)
	err := thread.Do(context.Background(), func() error {
		loadOnce.Do(func() { reference, _, loadErr = models.Load(dir) })
		if loadErr != nil {
			return loadErr
		}
		return mlx.Scoped(func() error { return fn(reference) })
	})
	if err != nil {
		t.Fatal(err)
	}
}

// goldenLayers loads testdata/qwen2/layers.safetensors inside the scope of
// the caller, which frees the arrays.
func goldenLayers() (map[string]*mlx.Array, error) {
	arrays, _, err := mlx.LoadSafetensors(modeltest.Testdata("qwen2", "layers.safetensors"))
	return arrays, err
}

// lastLogits returns the logits of the last position of a [1, L, vocab]
// array as [1, vocab].
func lastLogits(logits *mlx.Array) *mlx.Array {
	l, vocab := logits.Dim(1), logits.Dim(2)
	return mlx.Reshape(mlx.Slice(logits, []int{0, l - 1, 0}, []int{1, l, vocab}), 1, vocab)
}

// argmax returns the index of the largest element of a [1, vocab] array.
func argmax(a *mlx.Array) int32 {
	return mlx.AsType(mlx.Argmax(a, -1, false), mlx.Int32).Int32s()[0]
}

func TestForwardMatchesGoldenLogits(t *testing.T) {
	onModel(t, func(model models.Model) error {
		golden, err := goldenLayers()
		if err != nil {
			return err
		}
		defer free(golden)
		logits := model.Forward(golden["input_ids"], nil)
		if got, want := logits.Shape(), []int{1, 8, 151936}; !slices.Equal(got, want) {
			t.Errorf("logits have shape %v, want %v", got, want)
			return nil
		}
		last := lastLogits(logits)
		diff := modeltest.MaxAbsDiff(t, last, golden["logits_last"])
		t.Logf("logits_last: max abs diff %.3g", diff)
		if !(diff < 1e-1) {
			t.Errorf("last logits differ from the golden by %g, want below 1e-1", diff)
		}
		if got, want := argmax(last), argmax(golden["logits_last"]); got != want {
			t.Errorf("argmax of the last logits is %d, want %d", got, want)
		}
		return nil
	})
}

// free frees the arrays of a loaded file.
func free(arrays map[string]*mlx.Array) {
	for _, a := range arrays {
		a.Free()
	}
}

func TestStep0Top50MatchesGolden(t *testing.T) {
	cases, _ := modeltest.Generation(t)
	for i, c := range cases {
		onModel(t, func(model models.Model) error {
			prompt := mlx.FromInt32(c.PromptIDs, 1, len(c.PromptIDs))
			logits := mlx.AsType(lastLogits(model.Forward(prompt, nil)), mlx.Float32).Float32s()

			order := make([]int32, len(logits))
			for id := range order {
				order[id] = int32(id)
			}
			slices.SortStableFunc(order, func(a, b int32) int { return cmp.Compare(logits[b], logits[a]) })
			if order[0] != c.Top50IDs[0] {
				t.Errorf("case %d: the likeliest token is %d, want %d", i, order[0], c.Top50IDs[0])
			}
			shared := 0
			for _, id := range order[:50] {
				if slices.Contains(c.Top50IDs, id) {
					shared++
				}
			}
			worst := 0.0
			for j, id := range c.Top50IDs {
				worst = max(worst, math.Abs(float64(logits[id])-float64(c.Top50Logits[j])))
			}
			t.Logf("case %d: %d of our top 50 are in the golden top 50; max abs diff of the golden logits %.3g", i, shared, worst)
			if shared < 48 {
				t.Errorf("case %d: only %d of our top 50 tokens are in the golden top 50, want at least 48", i, shared)
			}
			if !(worst < 1e-1) {
				t.Errorf("case %d: the logits of the golden top 50 differ by %g, want below 1e-1", i, worst)
			}
			return nil
		})
	}
}

// feed runs the tokens through the model and the state in a scope of its own
// and returns the logits of the last position.
func feed(model models.Model, state models.State, ids ...int32) ([]float32, error) {
	var last []float32
	err := mlx.Scoped(func() error {
		logits := model.Forward(mlx.FromInt32(ids, 1, len(ids)), state)
		last = mlx.AsType(lastLogits(logits), mlx.Float32).Float32s()
		mlx.Keep(state.Arrays()...)
		return nil
	})
	return last, err
}

// The cache must reproduce what Python gets when it feeds the first seven
// golden tokens and then the eighth. That is not what one pass over all
// eight gives: float16 rounds differently on the two paths.
func TestIncrementalForwardMatchesGolden(t *testing.T) {
	onModel(t, func(model models.Model) error {
		golden, err := goldenLayers()
		if err != nil {
			return err
		}
		defer free(golden)
		want := golden["incremental_7_1_logits_last"]
		ids := golden["input_ids"].Int32s()

		state := model.NewState()
		defer state.Free()
		if _, err := feed(model, state, ids[:7]...); err != nil {
			return err
		}
		incremental, err := feed(model, state, ids[7])
		if err != nil {
			return err
		}

		got := mlx.FromFloat32(incremental, 1, len(incremental))
		diff := modeltest.MaxAbsDiff(t, got, want)
		t.Logf("incremental_7_1_logits_last: max abs diff %.3g", diff)
		if !(diff < 1e-1) {
			t.Errorf("the logits after 7+1 tokens differ from the golden by %g, want below 1e-1", diff)
		}
		if got, want := argmax(got), argmax(want); got != want {
			t.Errorf("argmax of the logits after 7+1 tokens is %d, want %d", got, want)
		}
		return nil
	})
}

func TestStateOffsetCountsTokens(t *testing.T) {
	onModel(t, func(model models.Model) error {
		state := model.NewState()
		defer state.Free()
		if got := state.Offset(); got != 0 {
			t.Errorf("a new state has offset %d, want 0", got)
		}
		for _, step := range []struct {
			ids  []int32
			want int
		}{
			{[]int32{151644, 8948, 198}, 3},
			{[]int32{2610}, 4},
			{[]int32{525, 1207}, 6},
		} {
			if _, err := feed(model, state, step.ids...); err != nil {
				return err
			}
			if got := state.Offset(); got != step.want {
				t.Errorf("after %d more tokens the offset is %d, want %d", len(step.ids), got, step.want)
			}
		}
		return nil
	})
}

func TestCacheGrowsPastOneAllocationStep(t *testing.T) {
	onModel(t, func(model models.Model) error {
		state := model.NewState()
		defer state.Free()
		var last []float32
		for i := range 300 {
			var err error
			if last, err = feed(model, state, 198); err != nil {
				return fmt.Errorf("step %d: %w", i, err)
			}
		}
		if got := state.Offset(); got != 300 {
			t.Errorf("after 300 single-token steps the offset is %d, want 300", got)
		}
		if slices.ContainsFunc(last, func(v float32) bool { return v != v }) {
			t.Error("the logits of step 300 contain NaN")
		}
		return nil
	})
}

func TestStateFreeReleasesMemory(t *testing.T) {
	const mib = 1 << 20
	cases, _ := modeltest.Generation(t)
	onModel(t, func(model models.Model) error {
		mlx.ClearCache()
		before := mlx.ActiveMemory()

		state := model.NewState()
		if _, err := feed(model, state, cases[0].PromptIDs...); err != nil {
			return err
		}
		mlx.ClearCache()
		held := mlx.ActiveMemory()
		state.Free()
		mlx.ClearCache()
		after := mlx.ActiveMemory()

		t.Logf("active memory: %d bytes before, %d with the state, %d after Free", before, held, after)
		if held < before+4*mib {
			t.Errorf("the state of %d tokens holds %d bytes, want several MiB", len(cases[0].PromptIDs), int64(held)-int64(before))
		}
		if diff := int64(after) - int64(before); diff > mib || diff < -mib {
			t.Errorf("active memory is %d bytes off the level before the state, want within 1 MiB", diff)
		}
		return nil
	})
}
