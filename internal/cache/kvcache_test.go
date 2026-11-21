package cache

import (
	"runtime"
	"slices"
	"testing"

	"github.com/bigshort2022/mlx-lm-go/mlx"
)

// Expected shapes and values in this file come from mlx_lm.models.cache.KVCache
// fed the same arrays.

// inScope pins the test and runs fn inside a scope that must not fail.
func inScope(t *testing.T, fn func()) {
	t.Helper()
	runtime.LockOSThread()
	t.Cleanup(runtime.UnlockOSThread)
	err := mlx.Scoped(func() error {
		fn()
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

// feed appends n positions to the cache: position p has keys (p, p) and
// values (-p, -p), for one head. It returns what the cache fetches.
func feed(c *KVCache, n int) (keys, values *mlx.Array) {
	k, v := make([]float32, 0, 2*n), make([]float32, 0, 2*n)
	for p := c.Offset(); p < c.Offset()+n; p++ {
		k = append(k, float32(p), float32(p))
		v = append(v, -float32(p), -float32(p))
	}
	return c.UpdateAndFetch(mlx.FromFloat32(k, 1, 1, n, 2), mlx.FromFloat32(v, 1, 1, n, 2))
}

// wantPositions fails the test unless keys and values hold positions 0 to
// n-1 as feed wrote them.
func wantPositions(t *testing.T, keys, values *mlx.Array, n int) {
	t.Helper()
	for name, a := range map[string]*mlx.Array{"keys": keys, "values": values} {
		if got, want := a.Shape(), []int{1, 1, n, 2}; !slices.Equal(got, want) {
			t.Fatalf("fetched %s have shape %v, want %v", name, got, want)
		}
		sign := float32(1)
		if name == "values" {
			sign = -1
		}
		for i, got := range a.Float32s() {
			if want := sign * float32(i/2); got != want {
				t.Fatalf("fetched %s[%d] = %v, want %v", name, i, got, want)
			}
		}
	}
}

// capacity returns the number of positions the buffers of c have room for.
func capacity(c *KVCache) int {
	return c.Arrays()[0].Dim(2)
}

func TestUpdateAndFetchReturnsEverythingSoFar(t *testing.T) {
	inScope(t, func() {
		c := &KVCache{}
		keys, values := feed(c, 2)
		wantPositions(t, keys, values, 2)
		keys, values = feed(c, 1)
		wantPositions(t, keys, values, 3)
		if c.Offset() != 3 {
			t.Errorf("Offset() = %d, want 3", c.Offset())
		}
	})
}

func TestEmptyCacheHasNoArrays(t *testing.T) {
	c := &KVCache{}
	if c.Offset() != 0 || len(c.Arrays()) != 0 {
		t.Errorf("empty cache has offset %d and %d arrays, want 0 and none", c.Offset(), len(c.Arrays()))
	}
	c.Free() // must not panic
}

func TestBuffersGrowInStepsOf256(t *testing.T) {
	tests := []struct {
		name         string
		chunks       []int
		wantCapacity int
	}{
		{"one position", []int{1}, 256},
		{"exactly one step", []int{256}, 256},
		{"more than one step at once", []int{300}, 512},
		{"a chunk that does not fit drops the unused room first", []int{100, 200}, 356},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inScope(t, func() {
				c := &KVCache{}
				var keys, values *mlx.Array
				total := 0
				for _, n := range tt.chunks {
					keys, values = feed(c, n)
					total += n
				}
				wantPositions(t, keys, values, total)
				if got := capacity(c); got != tt.wantCapacity {
					t.Errorf("the buffers have room for %d positions, want %d", got, tt.wantCapacity)
				}
			})
		})
	}
}

func TestSingleStepsGrowPastTheFirstAllocation(t *testing.T) {
	inScope(t, func() {
		c := &KVCache{}
		var keys, values *mlx.Array
		for range 256 {
			keys, values = feed(c, 1)
		}
		if got := capacity(c); got != 256 {
			t.Errorf("after 256 positions the buffers have room for %d, want 256", got)
		}
		wantPositions(t, keys, values, 256)
		keys, values = feed(c, 1)
		if got := capacity(c); got != 512 {
			t.Errorf("after 257 positions the buffers have room for %d, want 512", got)
		}
		wantPositions(t, keys, values, 257)
	})
}

// The reference rebinds self.keys on every update, which drops the buffer it
// replaces and lets MLX update in place. The cache must free it likewise,
// even when the buffer had been kept out of an earlier scope.
func TestUpdateFreesTheBuffersItReplaces(t *testing.T) {
	inScope(t, func() {
		c := &KVCache{}
		feed(c, 1)
		old := c.Arrays()
		feed(c, 1)
		for i, a := range old {
			if a.Valid() {
				t.Errorf("buffer %d of the previous update is still alive", i)
			}
		}
		for i, a := range c.Arrays() {
			if !a.Valid() {
				t.Errorf("buffer %d of the cache is freed", i)
			}
		}
	})
}

func TestFreeEmptiesTheCache(t *testing.T) {
	inScope(t, func() {
		c := &KVCache{}
		feed(c, 3)
		arrays := c.Arrays()
		c.Free()
		for i, a := range arrays {
			if a.Valid() {
				t.Errorf("buffer %d is still alive after Free", i)
			}
		}
		if c.Offset() != 0 || len(c.Arrays()) != 0 {
			t.Errorf("after Free the cache has offset %d and %d arrays, want 0 and none", c.Offset(), len(c.Arrays()))
		}
		keys, values := feed(c, 2)
		wantPositions(t, keys, values, 2)
	})
}

func TestLayers(t *testing.T) {
	inScope(t, func() {
		layers := NewLayers(3)
		if len(layers) != 3 || layers.Offset() != 0 || len(layers.Arrays()) != 0 {
			t.Fatalf("NewLayers(3) has %d caches, offset %d and %d arrays; want 3, 0 and none",
				len(layers), layers.Offset(), len(layers.Arrays()))
		}
		for _, c := range layers {
			feed(c, 2)
		}
		if layers.Offset() != 2 {
			t.Errorf("Offset() = %d, want 2", layers.Offset())
		}
		arrays := layers.Arrays()
		if len(arrays) != 6 {
			t.Fatalf("Arrays() has %d arrays, want keys and values of 3 layers", len(arrays))
		}
		layers.Free()
		for i, a := range arrays {
			if a.Valid() {
				t.Errorf("array %d is still alive after Free", i)
			}
		}
		if layers.Offset() != 0 {
			t.Errorf("Offset() = %d after Free, want 0", layers.Offset())
		}
	})
}
