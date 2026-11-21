package mlx

// #include <mlx/c/mlx.h>
import "C"

import "errors"

// Version returns the version of the MLX library this binary is linked to.
func Version() (string, error) {
	s := C.mlx_string_new()
	defer C.mlx_string_free(s)
	if C.mlx_version(&s) != 0 {
		return "", errors.New("mlx: mlx_version failed")
	}
	return C.GoString(C.mlx_string_data(s)), nil
}
