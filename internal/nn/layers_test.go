package nn

import (
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/bigshort2022/mlx-lm-go/mlx"
)

// Expected values in this file were computed with mlx.nn in Python on the
// same weights and inputs.

// wantFloats reports an error unless got is a float32 array of the given
// shape whose elements are within 1e-5 of want.
func wantFloats(t *testing.T, name string, got *mlx.Array, shape []int, want []float32) {
	t.Helper()
	if !slices.Equal(got.Shape(), shape) {
		t.Errorf("%s has shape %v, want %v", name, got.Shape(), shape)
		return
	}
	values := got.Float32s()
	for i := range want {
		if diff := math.Abs(float64(values[i]) - float64(want[i])); !(diff <= 1e-5) {
			t.Errorf("%s = %v, want %v", name, values, want)
			return
		}
	}
}

// denseLinear returns the tensors of a dense layer "fc" from 3 to 2 features.
func denseLinear() map[string]*mlx.Array {
	return map[string]*mlx.Array{
		"fc.weight": mlx.FromFloat32([]float32{1, 2, 3, -1, 0.5, 0.25}, 2, 3),
		"fc.bias":   mlx.FromFloat32([]float32{0.5, -0.5}, 2),
	}
}

func TestDenseLinearWithBias(t *testing.T) {
	inScope(t, func() {
		loader := NewLoader(denseLinear())
		linear := loader.Linear("fc", 3, true)
		if err := loader.Err(); err != nil {
			t.Fatal(err)
		}
		x := mlx.FromFloat32([]float32{1, 0, -1, 2, 2, 2}, 1, 2, 3)
		wantFloats(t, "linear(x)", linear.Forward(x), []int{1, 2, 2}, []float32{-1.5, -1.75, 12.5, -1})
	})
}

func TestDenseLinearWithoutBias(t *testing.T) {
	inScope(t, func() {
		weights := denseLinear()
		delete(weights, "fc.bias")
		loader := NewLoader(weights)
		linear := loader.Linear("fc", 3, false)
		if err := loader.Err(); err != nil {
			t.Fatal(err)
		}
		x := mlx.FromFloat32([]float32{1, 0, -1, 2, 2, 2}, 1, 2, 3)
		wantFloats(t, "linear(x)", linear.Forward(x), []int{1, 2, 2}, []float32{-2, -1.25, 12, -0.5})
	})
}

func TestLoaderReportsTheFirstMissingTensor(t *testing.T) {
	inScope(t, func() {
		weights := denseLinear()
		delete(weights, "fc.bias")
		loader := NewLoader(weights)
		loader.Linear("fc", 3, true)
		loader.RMSNorm("norm", 1e-6)
		err := loader.Err()
		if err == nil || !strings.Contains(err.Error(), `"fc.bias"`) {
			t.Errorf("Err() = %v, want an error naming fc.bias", err)
		}
	})
}

// quantizedShapes returns the tensors of a quantized layer "fc" whose packed
// weight has the given number of uint32 columns, with scales of the given
// type and number of columns. Only their shapes and types matter.
func quantizedShapes(weightColumns, scaleColumns int, scales mlx.DType) map[string]*mlx.Array {
	return map[string]*mlx.Array{
		"fc.weight": mlx.Zeros(mlx.Uint32, 2, weightColumns),
		"fc.scales": mlx.Zeros(scales, 2, scaleColumns),
		"fc.biases": mlx.Zeros(scales, 2, scaleColumns),
	}
}

// The packing of a quantized layer follows from the shapes of its tensors, as
// in mlx_lm.utils.infer_quant_config.
func TestLoaderInfersQuantizationFromShapes(t *testing.T) {
	tests := []struct {
		name                        string
		weightColumns, scaleColumns int
		scales                      mlx.DType
		wantGroupSize, wantBits     int
		wantMode                    string
	}{
		{"affine 4 bits", 8, 1, mlx.Float16, 64, 4, "affine"},
		{"affine 6 bits", 12, 2, mlx.Float16, 32, 6, "affine"},
		{"affine 8 bits", 16, 4, mlx.Float32, 16, 8, "affine"},
		{"nvfp4", 8, 4, mlx.Uint8, 16, 4, "nvfp4"},
		{"mxfp4", 8, 2, mlx.Uint8, 32, 4, "mxfp4"},
		{"mxfp8", 16, 2, mlx.Uint8, 32, 8, "mxfp8"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inScope(t, func() {
				loader := NewLoader(quantizedShapes(tt.weightColumns, tt.scaleColumns, tt.scales))
				m := loader.Linear("fc", 64, false).weight
				if err := loader.Err(); err != nil {
					t.Fatal(err)
				}
				if m.groupSize != tt.wantGroupSize || m.bits != tt.wantBits || m.mode != tt.wantMode {
					t.Errorf("inferred group size %d, %d bits, mode %q; want %d, %d, %q",
						m.groupSize, m.bits, m.mode, tt.wantGroupSize, tt.wantBits, tt.wantMode)
				}
			})
		})
	}
}

func TestLoaderRejectsPackingOfNoKnownMode(t *testing.T) {
	inScope(t, func() {
		loader := NewLoader(quantizedShapes(8, 1, mlx.Uint8)) // 4 bits in groups of 64 with integer scales
		loader.Linear("fc", 64, false)
		err := loader.Err()
		if err == nil || !strings.Contains(err.Error(), "fc") {
			t.Errorf("Err() = %v, want an error about the quantization of fc", err)
		}
	})
}

func TestLoaderListsUnusedTensors(t *testing.T) {
	inScope(t, func() {
		weights := denseLinear()
		weights["zz.weight"] = mlx.Zeros(mlx.Float32, 1)
		weights["aa.weight"] = mlx.Zeros(mlx.Float32, 1)
		loader := NewLoader(weights)
		loader.Linear("fc", 3, false)
		if got, want := loader.Unused(), []string{"aa.weight", "fc.bias", "zz.weight"}; !slices.Equal(got, want) {
			t.Errorf("Unused() = %v, want %v", got, want)
		}
	})
}

func TestDenseEmbedding(t *testing.T) {
	inScope(t, func() {
		loader := NewLoader(map[string]*mlx.Array{
			"emb.weight": mlx.FromFloat32([]float32{0, 1, 2, 3, 4, 5, 6, 7}, 4, 2),
		})
		embedding := loader.Embedding("emb", 2)
		ids := mlx.FromInt32([]int32{3, 0, 3}, 1, 3)
		wantFloats(t, "embedding(ids)", embedding.Forward(ids), []int{1, 3, 2}, []float32{6, 7, 0, 1, 6, 7})
		x := mlx.FromFloat32([]float32{1, -1, 0.5, 2}, 1, 2, 2)
		wantFloats(t, "embedding.as_linear(x)", embedding.AsLinear(x), []int{1, 2, 4},
			[]float32{-1, -1, -1, -1, 2, 7, 12, 17})
	})
}

func TestDenseRMSNorm(t *testing.T) {
	inScope(t, func() {
		loader := NewLoader(map[string]*mlx.Array{
			"norm.weight": mlx.FromFloat32([]float32{1, 2, 0.5, -1}, 4),
		})
		norm := loader.RMSNorm("norm", 1e-6)
		x := mlx.FromFloat32([]float32{1, 2, 3, 4, 0, 0, 0, 2}, 2, 4)
		wantFloats(t, "rms_norm(x)", norm.Forward(x), []int{2, 4},
			[]float32{0.365148, 1.46059, 0.547723, -1.46059, 0, 0, 0, -2})
	})
}

// eighths returns arange(16) / 8 as [1, 1, 2, 8]: one head, two positions.
func eighths() *mlx.Array {
	data := make([]float32, 16)
	for i := range data {
		data[i] = float32(i) / 8
	}
	return mlx.FromFloat32(data, 1, 1, 2, 8)
}

func TestRoPE(t *testing.T) {
	inScope(t, func() {
		rope := &RoPE{Dims: 8, Base: 1000000, Scale: 1}
		wantFloats(t, "rope(x)", rope.Forward(eighths(), 0), []int{1, 1, 2, 8}, []float32{
			0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875,
			-0.721904, 1.073059, 1.248249, 1.374941, 1.651924, 1.659757, 1.751249, 1.875044,
		})
		wantFloats(t, "rope(x, offset=5)", rope.Forward(eighths(), 5), []int{1, 1, 2, 8}, []float32{
			0.479462, 0.025031, 0.246247, 0.374862, 0.141831, 0.636886, 0.751241, 0.875059,
			1.379294, 0.798335, 1.239478, 1.374644, 1.16084, 1.808013, 1.757468, 1.875261,
		})
		scaled := &RoPE{Dims: 8, Base: 10000, Scale: 0.5}
		wantFloats(t, "rope(x, offset=3, scale=0.5)", scaled.Forward(eighths(), 3), []int{1, 1, 2, 8}, []float32{
			-0.498748, 0.030198, 0.238722, 0.373687, 0.035369, 0.636662, 0.753666, 0.875562,
			-1.780093, 0.779737, 1.214752, 1.371247, 0.285077, 1.816111, 1.774648, 1.877746,
		})
	})
}

func TestSwiGLU(t *testing.T) {
	inScope(t, func() {
		gate := mlx.FromFloat32([]float32{-2, -0.5, 0, 1, 3}, 5)
		x := mlx.FromFloat32([]float32{1, 2, 3, 4, -5}, 5)
		wantFloats(t, "swiglu(gate, x)", SwiGLU(gate, x), []int{5}, []float32{-0.238406, -0.377541, 0, 2.924234, -14.288611})
	})
}
