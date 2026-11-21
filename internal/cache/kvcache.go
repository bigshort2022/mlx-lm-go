// Package cache holds the key-value cache of the attention layers, a port of
// KVCache in mlx_lm/models/cache.py.
package cache

import "github.com/bigshort2022/mlx-lm-go/mlx"

// step is the number of positions the buffers grow by.
const step = 256

// KVCache keeps the keys and values of one attention layer for the positions
// processed so far. The zero value is an empty cache.
//
// Its buffers are created like any other array: inside a scope they belong to
// that scope, and whoever closes it must mlx.Keep(c.Arrays()...) first. The
// cache itself frees a buffer when it replaces it.
type KVCache struct {
	keys, values *mlx.Array // [B, heads, capacity, dims]; nil while empty
	offset       int        // positions in use
}

// Offset returns the number of positions in the cache.
func (c *KVCache) Offset() int { return c.offset }

// Arrays returns the buffers of the cache: nothing while it is empty.
func (c *KVCache) Arrays() []*mlx.Array {
	if c.keys == nil {
		return nil
	}
	return []*mlx.Array{c.keys, c.values}
}

// Free releases the buffers and leaves the cache empty.
func (c *KVCache) Free() {
	for _, a := range c.Arrays() {
		a.Free()
	}
	*c = KVCache{}
}

// UpdateAndFetch appends the keys and values of new positions, each of shape
// [B, heads, L, dims], and returns those of all positions so far. The cache
// keeps ownership of what it returns.
func (c *KVCache) UpdateAndFetch(keys, values *mlx.Array) (*mlx.Array, *mlx.Array) {
	prev := c.offset
	if c.keys == nil || prev+keys.Dim(2) > c.keys.Dim(2) {
		c.grow(keys, values)
	}
	c.offset += keys.Dim(2)
	c.keys = replace(c.keys, write(c.keys, keys, prev, c.offset))
	c.values = replace(c.values, write(c.values, values, prev, c.offset))
	if c.offset < c.keys.Dim(2) {
		return upTo(c.keys, c.offset), upTo(c.values, c.offset)
	}
	return c.keys, c.values
}

// grow makes room for the new positions in whole steps, dropping first the
// room a buffer has left when that is not a whole step.
func (c *KVCache) grow(keys, values *mlx.Array) {
	b, heads, n := keys.Dim(0), keys.Dim(1), keys.Dim(2)
	room := (step + n - 1) / step * step
	newKeys := mlx.Zeros(keys.DType(), b, heads, room, keys.Dim(3))
	newValues := mlx.Zeros(values.DType(), b, heads, room, values.Dim(3))
	if c.keys == nil {
		c.keys, c.values = newKeys, newValues
		return
	}
	oldKeys, oldValues := c.keys, c.values
	if c.offset%step != 0 {
		oldKeys, oldValues = upTo(oldKeys, c.offset), upTo(oldValues, c.offset)
	}
	c.keys = replace(c.keys, mlx.Concatenate(2, oldKeys, newKeys))
	c.values = replace(c.values, mlx.Concatenate(2, oldValues, newValues))
}

// replace frees a buffer of the cache and returns its successor. Dropping the
// last handle before the successor is evaluated is what lets MLX write into
// the old memory instead of copying it.
func replace(old, successor *mlx.Array) *mlx.Array {
	old.Free()
	return successor
}

// write returns buffer with positions from start up to stop set to update.
func write(buffer, update *mlx.Array, start, stop int) *mlx.Array {
	return mlx.SliceUpdate(buffer, update,
		[]int{0, 0, start, 0},
		[]int{buffer.Dim(0), buffer.Dim(1), stop, buffer.Dim(3)})
}

// upTo returns the first n positions of buffer.
func upTo(buffer *mlx.Array, n int) *mlx.Array {
	return mlx.Slice(buffer, []int{0, 0, 0, 0}, []int{buffer.Dim(0), buffer.Dim(1), n, buffer.Dim(3)})
}

// Layers is the caches of all layers of a model, first layer first.
type Layers []*KVCache

// NewLayers returns n empty caches.
func NewLayers(n int) Layers {
	layers := make(Layers, n)
	for i := range layers {
		layers[i] = &KVCache{}
	}
	return layers
}

// Offset returns the number of positions processed so far.
func (l Layers) Offset() int {
	if len(l) == 0 {
		return 0
	}
	return l[0].Offset()
}

// Arrays returns the buffers of every layer.
func (l Layers) Arrays() []*mlx.Array {
	var arrays []*mlx.Array
	for _, c := range l {
		arrays = append(arrays, c.Arrays()...)
	}
	return arrays
}

// Free releases the buffers of every layer and leaves the caches empty.
func (l Layers) Free() {
	for _, c := range l {
		c.Free()
	}
}
