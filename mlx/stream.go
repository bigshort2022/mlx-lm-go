package mlx

/*
#include <mlx/c/mlx.h>

// The calling thread's default GPU stream. MLX streams are thread affine, so
// every thread that uses the package has its own. The handle is created on
// first use and never freed, which saves a new/free pair on every operation
// and costs one small handle per thread.
static _Thread_local mlx_stream mlx_go_thread_gpu_stream;

static mlx_stream mlx_go_gpu_stream(void) {
	if (mlx_go_thread_gpu_stream.ctx == NULL) {
		mlx_go_thread_gpu_stream = mlx_default_gpu_stream_new();
	}
	return mlx_go_thread_gpu_stream;
}
*/
import "C"

// gpu returns the default GPU stream of the calling thread, which every
// operation runs on. The caller must not free it.
func gpu() C.mlx_stream {
	return C.mlx_go_gpu_stream()
}

// synchronize waits for the GPU work the calling thread has started. Until
// that work is done MLX holds on to its buffers, even those of arrays that
// were freed in the meantime.
func synchronize() {
	check(C.mlx_synchronize(gpu()))
}
