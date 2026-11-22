package generate

import (
	"slices"

	"github.com/bigshort2022/mlx-lm-go/internal/models"
	"github.com/bigshort2022/mlx-lm-go/mlx"
)

// stateCloner is implemented by models with resumable inference state.
// Each clone owns its handles; extending or freeing it leaves the source intact.
type stateCloner interface {
	CloneState(models.State) models.State
}

type promptCache struct {
	clone    stateCloner
	entries  []promptEntry // least recently used first
	maxSize  int
	maxBytes uint64
	bytes    uint64
}

type promptEntry struct {
	tokens []int32
	state  models.State
	bytes  uint64
}

// checkpoint leaves a short suffix uncached. Chat templates replace the end
// of the generation prompt when appending an answer or tool result, so the
// next turn usually shares more than this many tokens with the previous one.
func checkpoint(length int) int {
	return length - min(32, max(1, length/8))
}

// restore picks the longest saved prefix by comparing actual token IDs, not
// request IDs, text, or a hash. The caller excludes the final prompt token.
func (c *promptCache) restore(prefix []int32) (models.State, error) {
	best, length := -1, 0
	for i, entry := range c.entries {
		n := len(entry.tokens)
		if n > length && n <= len(prefix) && slices.Equal(entry.tokens, prefix[:n]) {
			best, length = i, n
		}
	}
	if best < 0 {
		return nil, nil
	}
	entry := c.entries[best]
	state, err := c.snapshot(entry.state)
	if err != nil {
		return nil, err
	}
	copy(c.entries[best:], c.entries[best+1:])
	c.entries[len(c.entries)-1] = entry
	return state, nil
}

// store saves an already evaluated state. Oversized states don't evict useful
// entries. Byte accounting includes the allocated tensor shapes and token IDs.
func (c *promptCache) store(prefix []int32, state models.State) error {
	n := uint64(len(prefix)) * 4
	for _, array := range state.Arrays() {
		n += array.NBytes()
	}
	if n > c.maxBytes {
		return nil
	}
	saved, err := c.snapshot(state)
	if err != nil {
		return err
	}
	for len(c.entries) >= c.maxSize || n > c.maxBytes-c.bytes {
		old := c.entries[0]
		old.state.Free()
		c.bytes -= old.bytes
		c.entries = slices.Delete(c.entries, 0, 1)
	}
	c.entries = append(c.entries, promptEntry{tokens: slices.Clone(prefix), state: saved, bytes: n})
	c.bytes += n
	return nil
}

func (c *promptCache) snapshot(src models.State) (dst models.State, err error) {
	err = mlx.Scoped(func() error {
		dst = c.clone.CloneState(src)
		mlx.Keep(dst.Arrays()...)
		return nil
	})
	return dst, err
}

func (c *promptCache) clear() {
	for _, entry := range c.entries {
		entry.state.Free()
	}
	c.entries, c.bytes = nil, 0
}
