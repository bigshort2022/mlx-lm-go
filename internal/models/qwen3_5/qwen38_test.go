package qwen3_5

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/bigshort2022/mlx-lm-go/internal/models"
	"github.com/bigshort2022/mlx-lm-go/internal/models/modeltest"
	"github.com/bigshort2022/mlx-lm-go/internal/tokenizer"
	"github.com/bigshort2022/mlx-lm-go/mlx"
)

type decoderProbe struct {
	Layer int         `json:"layer_index"`
	Unit  [][]float32 `json:"output_prefix"`
	Block [][]float32 `json:"layer_output_prefix"`
}

func TestQwen35DecoderParity(t *testing.T) {
	testDecoderProbe(t, "qwen35", "mlx-community/Qwen3.5-9B-MLX-4bit", "938d8919941c6e7efd3c7150eff7fe9d12afa631", mlx.Float32, 8192)
}

func TestQwen38DecoderParity(t *testing.T) {
	testDecoderProbe(t, "qwen38", "mlx-community/Qwen3.8-27B-4bit", "10c35caafbb80f7dc6a7a432cdd11af10a6d4818", mlx.BFloat16, 10240)
}

func testDecoderProbe(t *testing.T, fixtureName, modelID, revision string, aLogType mlx.DType, convChannels int) {
	t.Helper()
	if os.Getenv("MLX_LM_GO_TEST_QWEN38") != "1" {
		t.Skip("set MLX_LM_GO_TEST_QWEN38=1 to check the pinned Qwen snapshots")
	}
	modeltest.Pin(t)
	data, err := os.ReadFile(modeltest.Testdata("qwen38", fixtureName+".json"))
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		SchemaVersion int               `json:"schema_version"`
		ModelID       string            `json:"model_id"`
		Revision      string            `json:"revision"`
		Versions      map[string]string `json:"versions"`
		Probe         struct {
			Text      string       `json:"text"`
			PromptIDs []int32      `json:"prompt_ids"`
			Width     int          `json:"width"`
			Delta     decoderProbe `json:"delta"`
			Attention decoderProbe `json:"attention"`
		} `json:"layer_probe"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	probe := fixture.Probe
	if fixture.SchemaVersion != 1 || fixture.ModelID != modelID || fixture.Revision != revision ||
		fixture.Versions["mlx"] != "0.32.4" || fixture.Versions["mlx_lm"] != "0.32.2" ||
		len(probe.PromptIDs) != 10 || probe.Width != 8 || probe.Delta.Layer != 0 || probe.Attention.Layer != 3 {
		t.Fatal("unexpected oracle identity, versions, or layer probe shape")
	}
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
	dir := filepath.Join(root, "models--"+strings.ReplaceAll(modelID, "/", "--"), "snapshots", revision)
	tok, err := tokenizer.Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	ids := tok.Encode(probe.Text)
	if !slices.Equal(ids, probe.PromptIDs) {
		t.Fatalf("layer probe tokens = %v, want %v", ids, probe.PromptIDs)
	}
	base, _, err := models.Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	m := base.(*Model)
	defer m.Free()
	const prefix = textPrefix + "model.layers.0.linear_attn."
	if got := m.weights[prefix+"A_log"].DType(); got != aLogType {
		t.Errorf("A_log dtype = %v, want %v", got, aLogType)
	}
	if got := m.weights[prefix+"conv1d.weight"].Shape(); !slices.Equal(got, []int{convChannels, 4, 1}) {
		t.Errorf("conv1d shape = %v, want [%d 4 1]", got, convChannels)
	}
	err = mlx.Scoped(func() error {
		x := m.embed.Forward(mlx.FromInt32(ids, 1, len(ids)))
		for i := 0; i <= probe.Attention.Layer; i++ {
			l := &m.layers[i]
			a := l.attend(x, nil)
			x = l.finish(x, a)
			switch i {
			case probe.Delta.Layer:
				checkLayerProbe(t, "DeltaNet", a, x, probe.Delta, len(ids), probe.Width)
			case probe.Attention.Layer:
				checkLayerProbe(t, "full attention", a, x, probe.Attention, len(ids), probe.Width)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

func checkLayerProbe(t *testing.T, name string, unit, block *mlx.Array, want decoderProbe, length, width int) {
	t.Helper()
	for _, p := range []struct {
		name string
		a    *mlx.Array
		want [][]float32
	}{
		{"mixer output", unit, want.Unit}, {"decoder output", block, want.Block},
	} {
		if len(p.want) != length {
			t.Fatalf("%s %s: oracle has %d rows, want %d", name, p.name, len(p.want), length)
		}
		prefix := mlx.Slice(p.a, []int{0, 0, 0}, []int{1, length, width})
		got := mlx.AsType(prefix, mlx.Float32).Float32s()
		var worst float64
		for i, row := range p.want {
			if len(row) != width {
				t.Fatalf("%s %s: oracle row %d has %d columns, want %d", name, p.name, i, len(row), width)
			}
			for j, ref := range row {
				g := got[i*width+j]
				diff := math.Abs(float64(g - ref))
				worst = max(worst, diff)
				// BF16 absolute floor around cancellation near zero, plus two
				// units in the last place at the magnitude of an activation.
				_, exp := math.Frexp(math.Abs(float64(ref)))
				tol := 0.001 + 2*math.Ldexp(1, exp-8)
				if !(diff <= tol) {
					t.Errorf("%s %s [%d,%d] = %g, want %g (delta %g > %g)", name, p.name, i, j, g, ref, diff, tol)
				}
			}
		}
		t.Logf("%s %s max |delta| = %g", name, p.name, worst)
	}
}
