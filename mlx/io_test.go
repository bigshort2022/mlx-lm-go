package mlx

import (
	"errors"
	"path/filepath"
	"slices"
	"testing"
)

// fixturePath is written by testdata/gen_small.py with the Python reference.
const fixturePath = "testdata/small.safetensors"

func TestLoadSafetensorsReturnsEveryTensor(t *testing.T) {
	pin(t)
	arrays, metadata, err := LoadSafetensors(fixturePath)
	if err != nil {
		t.Fatalf("LoadSafetensors(%q): %v", fixturePath, err)
	}
	defer func() {
		for _, a := range arrays {
			a.Free()
		}
	}()

	type tensor struct {
		dtype DType
		shape []int
	}
	// The listing gen_small.py prints.
	want := map[string]tensor{
		"dtype.bfloat16":                     {BFloat16, []int{2}},
		"dtype.bool":                         {Bool, []int{2}},
		"dtype.complex64":                    {Complex64, []int{2}},
		"dtype.float16":                      {Float16, []int{2}},
		"dtype.float32":                      {Float32, []int{2}},
		"dtype.int16":                        {Int16, []int{2}},
		"dtype.int32":                        {Int32, []int{2}},
		"dtype.int64":                        {Int64, []int{2}},
		"dtype.int8":                         {Int8, []int{2}},
		"dtype.uint16":                       {Uint16, []int{2}},
		"dtype.uint32":                       {Uint32, []int{2}},
		"dtype.uint64":                       {Uint64, []int{2}},
		"dtype.uint8":                        {Uint8, []int{2}},
		"mxfp4.dequantize_expected":          {BFloat16, []int{8, 64}},
		"mxfp4.matmul_expected":              {Float32, []int{2, 8}},
		"mxfp4.scales":                       {Uint8, []int{8, 2}},
		"mxfp4.w_q":                          {Uint32, []int{8, 8}},
		"quant.biases":                       {Float32, []int{8, 1}},
		"quant.dequantize_expected":          {Float32, []int{8, 64}},
		"quant.matmul_expected":              {Float32, []int{2, 8}},
		"quant.matmul_untransposed_expected": {Float32, []int{2, 64}},
		"quant.scales":                       {Float32, []int{8, 1}},
		"quant.w":                            {Float32, []int{8, 64}},
		"quant.w_q":                          {Uint32, []int{8, 8}},
		"quant.x":                            {Float32, []int{2, 64}},
		"quant.y":                            {Float32, []int{2, 8}},
		"rms.expected":                       {Float32, []int{2, 4}},
		"rms.weight":                         {Float32, []int{4}},
		"rms.x":                              {Float32, []int{2, 4}},
		"rope.expected_offset0":              {Float32, []int{1, 2, 4, 8}},
		"rope.expected_offset3":              {Float32, []int{1, 2, 4, 8}},
		"rope.expected_traditional":          {Float32, []int{1, 2, 4, 8}},
		"rope.x":                             {Float32, []int{1, 2, 4, 8}},
		"sdpa.expected_array":                {Float32, []int{1, 2, 4, 8}},
		"sdpa.expected_causal":               {Float32, []int{1, 2, 4, 8}},
		"sdpa.expected_unmasked":             {Float32, []int{1, 2, 4, 8}},
		"sdpa.k":                             {Float32, []int{1, 2, 4, 8}},
		"sdpa.mask":                          {Float32, []int{4, 4}},
		"sdpa.q":                             {Float32, []int{1, 2, 4, 8}},
		"sdpa.v":                             {Float32, []int{1, 2, 4, 8}},
	}
	if len(arrays) != len(want) {
		t.Errorf("loaded %d tensors, want %d", len(arrays), len(want))
	}
	for name, w := range want {
		a, ok := arrays[name]
		if !ok {
			t.Errorf("tensor %q is missing", name)
			continue
		}
		if a.DType() != w.dtype || !slices.Equal(a.Shape(), w.shape) {
			t.Errorf("tensor %q is %v %v, want %v %v", name, a.DType(), a.Shape(), w.dtype, w.shape)
		}
	}
	if got := metadata["generator"]; got != "gen_small.py" {
		t.Errorf("metadata[\"generator\"] = %q, want \"gen_small.py\" (metadata: %v)", got, metadata)
	}
}

func TestLoadSafetensorsReturnsErrorForMissingFile(t *testing.T) {
	pin(t)
	path := filepath.Join(t.TempDir(), "missing.safetensors")
	arrays, metadata, err := LoadSafetensors(path)
	if err == nil {
		t.Fatal("LoadSafetensors of a missing file returned no error")
	}
	var mlxErr *Error
	if !errors.As(err, &mlxErr) || mlxErr.Msg == "" {
		t.Errorf("error is %T (%v), want *Error with the MLX message", err, err)
	}
	if arrays != nil || metadata != nil {
		t.Errorf("returned arrays %v and metadata %v alongside the error, want nil", arrays, metadata)
	}
}

func TestLoadedArraysOutliveEnclosingScope(t *testing.T) {
	pin(t)
	var arrays map[string]*Array
	err := Scoped(func() error {
		var err error
		arrays, _, err = LoadSafetensors(fixturePath)
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		for _, a := range arrays {
			a.Free()
		}
	}()
	for name, a := range arrays {
		if !a.Valid() {
			t.Errorf("tensor %q was freed by the enclosing scope, want it owned by the caller", name)
		}
	}
	// rms.weight is [0.5, 1.0, 1.5, 2.0] in gen_small.py.
	if got := arrays["rms.weight"].Float32s(); !slices.Equal(got, []float32{0.5, 1, 1.5, 2}) {
		t.Errorf("rms.weight = %v after the scope, want [0.5 1 1.5 2]", got)
	}
}
