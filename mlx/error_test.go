package mlx

import (
	"errors"
	"testing"
)

func TestShapeMismatchReturnsErrorFromScoped(t *testing.T) {
	pin(t)
	err := Scoped(func() error {
		a := FromFloat32([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
		b := FromFloat32([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
		Eval(Matmul(a, b))
		return nil
	})
	if err == nil {
		t.Fatal("Scoped returned nil for a [2,3] x [2,3] matmul, want an error")
	}
	var mlxErr *Error
	if !errors.As(err, &mlxErr) {
		t.Fatalf("Scoped returned %T (%v), want *Error", err, err)
	}
	if mlxErr.Msg == "" {
		t.Fatal("Error.Msg is empty, want the MLX message")
	}
}
