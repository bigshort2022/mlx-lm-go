package mlx

import "fmt"

// DType is the element type of an Array. It mirrors mlx_dtype, in the same
// order.
type DType int

// The element types MLX supports.
const (
	Bool DType = iota
	Uint8
	Uint16
	Uint32
	Uint64
	Int8
	Int16
	Int32
	Int64
	Float16
	Float32
	Float64
	BFloat16
	Complex64
)

var dtypeNames = [...]string{
	Bool: "bool", Uint8: "uint8", Uint16: "uint16", Uint32: "uint32", Uint64: "uint64",
	Int8: "int8", Int16: "int16", Int32: "int32", Int64: "int64",
	Float16: "float16", Float32: "float32", Float64: "float64", BFloat16: "bfloat16", Complex64: "complex64",
}

// String returns the MLX name of the type, such as "float32".
func (d DType) String() string {
	if d < 0 || int(d) >= len(dtypeNames) {
		return fmt.Sprintf("DType(%d)", int(d))
	}
	return dtypeNames[d]
}
