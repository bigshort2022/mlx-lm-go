package generate

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/bigshort2022/mlx-lm-go/internal/chattemplate"
	"github.com/bigshort2022/mlx-lm-go/internal/models/modeltest"
	"github.com/bigshort2022/mlx-lm-go/internal/sample"
	"github.com/bigshort2022/mlx-lm-go/internal/tokenizer"
	"github.com/bigshort2022/mlx-lm-go/mlx"
)

type qwenOracle struct {
	SchemaVersion    int               `json:"schema_version"`
	ModelID          string            `json:"model_id"`
	Revision         string            `json:"revision"`
	Versions         map[string]string `json:"versions"`
	EOS              []int32           `json:"eos_token_ids"`
	TopK             int               `json:"top_k"`
	MaxGreedyTokens  int               `json:"max_greedy_tokens"`
	NearTieThreshold float32           `json:"near_tie_threshold"`
	Cases            []qwenOracleCase  `json:"cases"`
}

type qwenOracleCase struct {
	Name           string                     `json:"name"`
	Messages       json.RawMessage            `json:"messages"`
	TemplateKwargs map[string]json.RawMessage `json:"template_kwargs"`
	RenderedPrompt string                     `json:"rendered_prompt"`
	PromptIDs      []int32                    `json:"prompt_ids"`
	Runs           []qwenOracleRun            `json:"runs"`
}

type qwenOracleRun struct {
	PrefillStep  int       `json:"prefill_step_size"`
	TopIDs       []int32   `json:"top_ids"`
	TopLogits    []float32 `json:"top_logits"`
	GreedyIDs    []int32   `json:"greedy_ids"`
	GreedyMargin []float32 `json:"greedy_margins"`
	StopReason   string    `json:"stop_reason"`
}

func TestQwen35GenerationParity(t *testing.T) {
	testQwenOracle(t, "qwen35", "mlx-community/Qwen3.5-9B-MLX-4bit", "938d8919941c6e7efd3c7150eff7fe9d12afa631", []int32{248046})
}

func TestQwen38GenerationParity(t *testing.T) {
	testQwenOracle(t, "qwen38", "mlx-community/Qwen3.8-27B-4bit", "10c35caafbb80f7dc6a7a432cdd11af10a6d4818", []int32{248044, 248046})
}

func testQwenOracle(t *testing.T, fixtureName, modelID, revision string, eosIDs []int32) {
	t.Helper()
	if os.Getenv("MLX_LM_GO_TEST_QWEN38") != "1" {
		t.Skip("set MLX_LM_GO_TEST_QWEN38=1 to check the pinned Qwen snapshots")
	}
	closeReference()
	fixture, err := os.ReadFile(modeltest.Testdata("qwen38", fixtureName+".json"))
	if err != nil {
		t.Fatal(err)
	}
	var oracle qwenOracle
	if err := json.Unmarshal(fixture, &oracle); err != nil {
		t.Fatal(err)
	}
	if oracle.SchemaVersion != 1 || oracle.ModelID != modelID || oracle.Revision != revision ||
		oracle.Versions["mlx"] != "0.32.4" || oracle.Versions["mlx_lm"] != "0.32.2" ||
		!slices.Equal(oracle.EOS, eosIDs) || oracle.TopK != 10 || oracle.MaxGreedyTokens != 6 ||
		oracle.NearTieThreshold != 0.125 || len(oracle.Cases) != 4 {
		t.Fatal("unexpected oracle identity, versions, EOS, or fixture constants")
	}
	version, err := mlx.Version()
	if err != nil || version != oracle.Versions["mlx"] {
		t.Fatalf("native MLX version = %q, err = %v; want %s", version, err, oracle.Versions["mlx"])
	}
	dir := pinnedQwenSnapshot(t, modelID, revision)
	tok, err := tokenizer.Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	_, fallback, ok := tok.EOSToken()
	if !ok || fallback != 248046 {
		t.Fatalf("tokenizer EOS = %d (present = %t), want 248046", fallback, ok)
	}
	tmpl, err := chattemplate.Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	e, err := Load(dir, Options{EOSTokenIDs: []int32{fallback}})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	gotEOS := slices.Clone(e.eos)
	slices.Sort(gotEOS)
	if !slices.Equal(gotEOS, oracle.EOS) {
		t.Fatalf("effective EOS = %v, want %v", gotEOS, oracle.EOS)
	}
	for _, c := range oracle.Cases {
		t.Run(c.Name, func(t *testing.T) {
			var addGenerationPrompt bool
			if err := json.Unmarshal(c.TemplateKwargs["add_generation_prompt"], &addGenerationPrompt); err != nil || !addGenerationPrompt {
				t.Fatalf("add_generation_prompt = %t (%v), want true", addGenerationPrompt, err)
			}
			delete(c.TemplateKwargs, "add_generation_prompt")
			kwargs, err := json.Marshal(c.TemplateKwargs)
			if err != nil {
				t.Fatal(err)
			}
			prompt, err := tmpl.Render(chattemplate.Input{Messages: c.Messages, Kwargs: kwargs, AddGenerationPrompt: addGenerationPrompt})
			if err != nil || prompt != c.RenderedPrompt {
				t.Fatalf("render = %q (%v), want %q", prompt, err, c.RenderedPrompt)
			}
			ids := tok.Encode(prompt)
			if len(ids) < 5 || !slices.Equal(ids, c.PromptIDs) {
				t.Fatalf("prompt IDs = %v, want %v", ids, c.PromptIDs)
			}
			if len(c.Runs) != 2 || c.Runs[0].PrefillStep != 2048 || c.Runs[1].PrefillStep != 4 {
				t.Fatal("oracle must cover prefill steps 2048 and 4")
			}
			for _, golden := range c.Runs {
				t.Run(fmt.Sprint(golden.PrefillStep), func(t *testing.T) {
					e.prefillStepSize = golden.PrefillStep
					checkQwenLogits(t, qwenPromptLogits(t, e, ids), golden, oracle.TopK, oracle.NearTieThreshold)
					checkQwenGreedy(t, e, ids, golden, oracle.MaxGreedyTokens)
				})
			}
		})
	}
}

func pinnedQwenSnapshot(t *testing.T, id, revision string) string {
	t.Helper()
	root := os.Getenv("HF_HUB_CACHE")
	if root == "" {
		home := os.Getenv("HF_HOME")
		if home == "" {
			userHome, err := os.UserHomeDir()
			if err != nil {
				t.Fatal(err)
			}
			home = filepath.Join(userHome, ".cache", "huggingface")
		}
		root = filepath.Join(home, "hub")
	}
	dir := filepath.Join(root, "models--"+strings.ReplaceAll(id, "/", "--"), "snapshots", revision)
	files := []string{"config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "model.safetensors.index.json"}
	if id == "mlx-community/Qwen3.8-27B-4bit" {
		files = append(files, "generation_config.json")
	}
	for _, name := range files {
		if _, err := os.Stat(filepath.Join(dir, name)); err != nil {
			t.Fatalf("pinned snapshot %s: %v", id, err)
		}
	}
	return dir
}

// qwenPromptLogits reuses the actual engine prefill steps to measure the raw
// logits that enter greedy sampling (the last prompt token is always separate).
func qwenPromptLogits(t *testing.T, e *Engine, ids []int32) []float32 {
	t.Helper()
	var result []float32
	err := e.thread.Do(context.Background(), func() error {
		req := greedy(ids, 1)
		sampler, err := sample.New(req.Sampling)
		if err != nil {
			return err
		}
		r := run{model: e.model, state: e.model.NewState(), sampler: sampler}
		defer r.free()
		for len(ids) > 1 {
			n := min(e.prefillStepSize, len(ids)-1)
			if err := r.prefill(ids[:n]); err != nil {
				return err
			}
			ids = ids[n:]
			mlx.ClearCache()
		}
		return mlx.Scoped(func() error {
			z := r.model.Forward(mlx.FromInt32(ids, 1, 1), r.state)
			if shape := z.Shape(); !slices.Equal(shape, []int{1, 1, e.vocab}) {
				return fmt.Errorf("last logits shape = %v, want [1 1 %d]", shape, e.vocab)
			}
			result = mlx.AsType(z, mlx.Float32).Float32s()
			return nil
		})
	})
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func checkQwenLogits(t *testing.T, logits []float32, golden qwenOracleRun, topK int, nearTie float32) {
	t.Helper()
	if len(golden.TopIDs) != topK || len(golden.TopLogits) != topK {
		t.Fatal("oracle is missing its top logits")
	}
	var maxDelta float32
	for i, id := range golden.TopIDs {
		if id < 0 || int(id) >= len(logits) || !finiteQwen(golden.TopLogits[i]) {
			t.Fatalf("invalid oracle score or ID at rank %d", i)
		}
		want, got := golden.TopLogits[i], logits[id]
		d := float32(math.Abs(float64(got - want)))
		maxDelta = max(maxDelta, d)
		// Two BF16 ULPs at the oracle score (0.25 for scores in [16,32)).
		_, exponent := math.Frexp(math.Abs(float64(want)))
		if tol := float32(2 * math.Ldexp(1, exponent-8)); d > tol {
			t.Errorf("token %d: raw logit = %g, want %g (delta %g > %g)", id, got, want, d, tol)
		}
	}
	top := make([]int32, 0, topK+1)
	for idx, score := range logits {
		if !finiteQwen(score) {
			t.Fatalf("token %d has nonfinite logit %g", idx, score)
		}
		if len(top) == topK && score <= logits[top[topK-1]] {
			continue
		}
		i := len(top)
		for i > 0 && score > logits[top[i-1]] {
			i--
		}
		top = slices.Insert(top, i, int32(idx))
		if len(top) > topK {
			top = top[:topK]
		}
	}
	t.Logf("max |raw logit delta| = %g; top IDs Go %v Python %v", maxDelta, top, golden.TopIDs)
	// Ignore rank-only reversals where either side is BF16-near-tied.
	// The separate greedy test is exact and does not use this allowance.
	for i, id := range golden.TopIDs {
		for j, below := range golden.TopIDs[i+1:] {
			if golden.TopLogits[i]-golden.TopLogits[i+j+1] > nearTie && logits[below]-logits[id] > nearTie {
				t.Errorf("well-separated raw IDs %d and %d reversed rank", id, below)
			}
		}
		if slices.Contains(top, id) {
			continue
		}
		for _, extra := range top {
			if !slices.Contains(golden.TopIDs, extra) &&
				(golden.TopLogits[i]-golden.TopLogits[topK-1] > nearTie || logits[extra]-logits[id] > nearTie) {
				t.Errorf("top-%d lost token %d to non-near-tied token %d", topK, id, extra)
			}
		}
	}
}

func checkQwenGreedy(t *testing.T, e *Engine, prompt []int32, golden qwenOracleRun, budget int) {
	t.Helper()
	want := golden.GreedyIDs
	if len(want) == 0 || len(want) > budget || len(golden.GreedyMargin) != len(want) {
		t.Fatal("invalid oracle greedy sequence/margins")
	}
	wantFinish := "length"
	if slices.Contains(e.eos, want[len(want)-1]) {
		if golden.StopReason != "eos" {
			t.Fatal("oracle EOS token and stop reason disagree")
		}
		want, wantFinish = want[:len(want)-1], "stop"
	} else if golden.StopReason != "length" || len(want) != budget {
		t.Fatal("oracle did not end at EOS or exhaust its budget")
	}
	got, result := generate(t, e, greedy(prompt, budget))
	if !slices.Equal(got, want) {
		t.Errorf("greedy IDs = %v, want %v (oracle margins %v)", got, want, golden.GreedyMargin)
	}
	if result.PromptTokens != len(prompt) || result.CompletionTokens != len(golden.GreedyIDs) || result.CachedTokens != 0 || result.FinishReason != wantFinish {
		t.Errorf("usage = %+v, want prompt=%d completion=%d cached=0 finish=%s", result, len(prompt), len(golden.GreedyIDs), wantFinish)
	}
}

func finiteQwen(x float32) bool {
	return !math.IsNaN(float64(x)) && !math.IsInf(float64(x), 0)
}
