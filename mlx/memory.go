package mlx

// #include <mlx/c/mlx.h>
import "C"

// This file reports and limits the memory of MLX's allocator. All sizes are
// in bytes.

// ActiveMemory returns the memory held by live arrays. It excludes the buffer
// cache.
func ActiveMemory() uint64 {
	var n C.size_t
	check(C.mlx_get_active_memory(&n))
	return uint64(n)
}

// PeakMemory returns the largest ActiveMemory since the process started or
// since the last ResetPeakMemory.
func PeakMemory() uint64 {
	var n C.size_t
	check(C.mlx_get_peak_memory(&n))
	return uint64(n)
}

// CacheMemory returns the memory of freed buffers that MLX keeps for reuse.
func CacheMemory() uint64 {
	var n C.size_t
	check(C.mlx_get_cache_memory(&n))
	return uint64(n)
}

// ResetPeakMemory sets PeakMemory back to zero.
func ResetPeakMemory() {
	check(C.mlx_reset_peak_memory())
}

// ClearCache returns the cached buffers to the system. It first waits for the
// GPU work in flight on the calling thread, so that the buffers of arrays
// freed just before the call are returned too.
func ClearCache() {
	synchronize()
	check(C.mlx_clear_cache())
}

// SetCacheLimit sets how much memory the buffer cache may hold and returns
// the previous limit. With a limit of zero freed buffers are not cached.
func SetCacheLimit(limit uint64) (previous uint64) {
	var old C.size_t
	check(C.mlx_set_cache_limit(&old, C.size_t(limit)))
	return uint64(old)
}

// SetMemoryLimit sets how much memory MLX uses before it starts waiting for
// buffers to be freed, and returns the previous limit.
func SetMemoryLimit(limit uint64) (previous uint64) {
	var old C.size_t
	check(C.mlx_set_memory_limit(&old, C.size_t(limit)))
	return uint64(old)
}

// SetWiredLimit sets how much of MLX's memory the system keeps wired, that
// is resident and not compressed or swapped, and returns the previous limit.
// Zero disables wiring.
func SetWiredLimit(limit uint64) (previous uint64) {
	var old C.size_t
	check(C.mlx_set_wired_limit(&old, C.size_t(limit)))
	return uint64(old)
}
