package mlx

import "testing"

const mib = 1 << 20

// sixteenMiB returns an evaluated float32 array that occupies 16 MiB.
func sixteenMiB() *Array {
	a := Zeros(Float32, 4096, 1024)
	Eval(a)
	return a
}

func TestActiveMemoryFollowsArrayLifetime(t *testing.T) {
	pin(t)
	ClearCache()
	before := ActiveMemory()
	a := sixteenMiB()
	held := ActiveMemory()
	if held < before+16*mib {
		t.Errorf("ActiveMemory() went from %d to %d after evaluating a 16 MiB array, want it to grow by 16 MiB", before, held)
	}
	a.Free()
	ClearCache()
	if after := ActiveMemory(); after > before+mib {
		t.Errorf("ActiveMemory() = %d after Free and ClearCache, want it back near %d", after, before)
	}
}

func TestCacheMemoryHoldsFreedBuffersUntilClearCache(t *testing.T) {
	pin(t)
	ClearCache()
	sixteenMiB().Free()
	// The buffer reaches the cache once the GPU work that produced it is done.
	synchronize()
	if cached := CacheMemory(); cached < 16*mib {
		t.Errorf("CacheMemory() = %d after freeing a 16 MiB array, want the buffer kept in the cache", cached)
	}
	ClearCache()
	if cached := CacheMemory(); cached != 0 {
		t.Errorf("CacheMemory() = %d after ClearCache, want 0", cached)
	}
}

func TestPeakMemoryIsHighWaterMarkUntilReset(t *testing.T) {
	pin(t)
	a := sixteenMiB()
	if peak, active := PeakMemory(), ActiveMemory(); peak < active {
		t.Errorf("PeakMemory() = %d is below ActiveMemory() = %d", peak, active)
	}
	a.Free()
	ClearCache()
	high := PeakMemory()
	if high < 16*mib {
		t.Errorf("PeakMemory() = %d after a 16 MiB array was freed, want it to remember the high-water mark", high)
	}
	ResetPeakMemory()
	// Python: mx.reset_peak_memory(); mx.get_peak_memory() == 0.
	if low := PeakMemory(); low != 0 {
		t.Errorf("PeakMemory() = %d after ResetPeakMemory, want 0", low)
	}
}

func TestSetCacheLimitReturnsPreviousLimit(t *testing.T) {
	previous := SetCacheLimit(0)
	if previous == 0 {
		t.Error("SetCacheLimit(0) returned 0, want MLX's non-zero default limit")
	}
	if got := SetCacheLimit(previous); got != 0 {
		t.Errorf("restoring the cache limit returned %d, want the 0 set before", got)
	}
}

// Python: mx.set_cache_limit and mx.set_memory_limit return the same value in
// a fresh process, because the cache limit defaults to the memory limit.
func TestDefaultCacheLimitIsTheMemoryLimit(t *testing.T) {
	cache := SetCacheLimit(0)
	SetCacheLimit(cache)
	memory := SetMemoryLimit(1 << 30)
	SetMemoryLimit(memory)
	if cache != memory {
		t.Errorf("default cache limit is %d and default memory limit is %d, want them equal", cache, memory)
	}
}

func TestSetMemoryLimitReturnsPreviousLimit(t *testing.T) {
	const limit = 1 << 30
	previous := SetMemoryLimit(limit)
	if previous == 0 {
		t.Error("SetMemoryLimit returned 0, want MLX's non-zero default limit")
	}
	if got := SetMemoryLimit(previous); got != limit {
		t.Errorf("restoring the memory limit returned %d, want the %d set before", got, limit)
	}
}

func TestSetWiredLimitReturnsPreviousLimit(t *testing.T) {
	const limit = 1 << 30
	// Python: mx.set_wired_limit(...) returns 0 in a fresh process.
	previous := SetWiredLimit(limit)
	if previous != 0 {
		t.Errorf("SetWiredLimit returned %d, want MLX's default of 0", previous)
	}
	if got := SetWiredLimit(previous); got != limit {
		t.Errorf("restoring the wired limit returned %d, want the %d set before", got, limit)
	}
}
