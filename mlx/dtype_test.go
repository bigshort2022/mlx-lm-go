package mlx

import "testing"

// The names are what Python prints for each type, without the "mlx.core."
// prefix.
func TestDTypeStringMatchesPythonName(t *testing.T) {
	names := map[DType]string{
		Bool: "bool", Uint8: "uint8", Uint16: "uint16", Uint32: "uint32", Uint64: "uint64",
		Int8: "int8", Int16: "int16", Int32: "int32", Int64: "int64",
		Float16: "float16", Float32: "float32", Float64: "float64", BFloat16: "bfloat16", Complex64: "complex64",
	}
	for dtype, want := range names {
		if got := dtype.String(); got != want {
			t.Errorf("DType(%d).String() = %q, want %q", int(dtype), got, want)
		}
	}
	if got, want := DType(99).String(), "DType(99)"; got != want {
		t.Errorf("DType(99).String() = %q, want %q", got, want)
	}
}

// The fixture holds one tensor per type, named by Python. Loading them shows
// that each DType constant has the number MLX gives that type. float64 is not
// in the file because MLX cannot write it; its neighbours pin it.
func TestDTypeConstantsMatchMLXNumbering(t *testing.T) {
	pin(t)
	arrays, _, err := LoadSafetensors(fixturePath)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		for _, a := range arrays {
			a.Free()
		}
	}()
	for _, want := range []DType{
		Bool, Uint8, Uint16, Uint32, Uint64, Int8, Int16, Int32, Int64, Float16, Float32, BFloat16, Complex64,
	} {
		name := "dtype." + want.String()
		a, ok := arrays[name]
		if !ok {
			t.Errorf("fixture has no tensor %q", name)
			continue
		}
		if got := a.DType(); got != want {
			t.Errorf("tensor %q has DType %d (%v), want %d", name, int(got), got, int(want))
		}
	}
}
