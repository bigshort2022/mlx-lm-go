package mlx

/*
#include <stdlib.h>
#include <string.h>
#include <mlx/c/mlx.h>

// The message of the most recent MLX error that Go has not collected yet, or
// NULL. A plain global, because the package is single-threaded by contract.
static char *mlx_go_pending_error;

// Replaces mlx-c's default handler, which prints the message and calls exit.
static void mlx_go_record_error(const char *msg, void *data) {
	free(mlx_go_pending_error);
	mlx_go_pending_error = strdup(msg);
}

static void mlx_go_install_error_handler(void) {
	mlx_set_error_handler(mlx_go_record_error, NULL, NULL);
}

// Hands the pending message to the caller, who must free it.
static char *mlx_go_take_error(void) {
	char *msg = mlx_go_pending_error;
	mlx_go_pending_error = NULL;
	return msg;
}
*/
import "C"

import "unsafe"

// Error is an error reported by MLX.
type Error struct{ Msg string }

func (e *Error) Error() string { return "mlx: " + e.Msg }

func init() {
	C.mlx_go_install_error_handler()
}

// takeError returns the error MLX recorded for the call that just failed.
func takeError() *Error {
	msg := C.mlx_go_take_error()
	if msg == nil {
		return &Error{Msg: "call failed without a message"}
	}
	defer C.free(unsafe.Pointer(msg))
	return &Error{Msg: C.GoString(msg)}
}

// check panics with the recorded *Error if an MLX call returned failure.
func check(rc C.int) {
	if rc != 0 {
		panic(takeError())
	}
}
