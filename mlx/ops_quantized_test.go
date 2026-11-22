package mlx

import "testing"

// The inputs and expected values of these tests are in the fixture: an
// [8,64] matrix quantized by Python with mx.quantize, and what Python's
// mx.quantized_matmul and mx.dequantize return for it.

const quantTolerance = 1e-4

func TestQuantizedMatmul(t *testing.T) {
	f := fixture(t)
	inScope(t, func() {
		got := QuantizedMatmul(f["quant.x"], f["quant.w_q"], f["quant.scales"], f["quant.biases"], true, 64, 4, "affine")
		wantFloats(t, "quantized_matmul(transpose)", got, []int{2, 8}, f["quant.matmul_expected"].Float32s(), quantTolerance)

		got = QuantizedMatmul(f["quant.y"], f["quant.w_q"], f["quant.scales"], f["quant.biases"], false, 64, 4, "affine")
		wantFloats(t, "quantized_matmul(no transpose)", got, []int{2, 64},
			f["quant.matmul_untransposed_expected"].Float32s(), quantTolerance)
	})
}

func TestQuantizedMatmulWithoutBiases(t *testing.T) {
	f := fixture(t)
	inScope(t, func() {
		got := QuantizedMatmul(f["quant.x"], f["mxfp4.w_q"], f["mxfp4.scales"], nil, true, 32, 4, "mxfp4")
		wantFloats(t, "quantized_matmul(mxfp4)", got, []int{2, 8}, f["mxfp4.matmul_expected"].Float32s(), quantTolerance)
	})
}

func TestDequantize(t *testing.T) {
	f := fixture(t)
	inScope(t, func() {
		got := Dequantize(f["quant.w_q"], f["quant.scales"], f["quant.biases"], 64, 4, "affine")
		wantFloats(t, "dequantize", got, []int{8, 64}, f["quant.dequantize_expected"].Float32s(), quantTolerance)
	})
}

func TestDequantizeWithoutBiases(t *testing.T) {
	f := fixture(t)
	inScope(t, func() {
		// mxfp4 dequantizes to bfloat16, in Python too.
		got := Dequantize(f["mxfp4.w_q"], f["mxfp4.scales"], nil, 32, 4, "mxfp4")
		if got.DType() != BFloat16 {
			t.Errorf("Dequantize(mxfp4) returned %v, want bfloat16", got.DType())
		}
		want := AsType(f["mxfp4.dequantize_expected"], Float32).Float32s()
		wantFloats(t, "dequantize(mxfp4)", AsType(got, Float32), []int{8, 64}, want, quantTolerance)
	})
}
