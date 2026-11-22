package tokenizer

import (
	"fmt"
	"unicode/utf8"
)

// byteRunes is the byte-level alphabet: byteRunes[b] is the character that
// stands for byte b in vocabulary tokens. Printable bytes stand for
// themselves; the others take the characters from U+0100 on, in byte order,
// which is what makes a space "Ġ".
var byteRunes = func() (alphabet [256]rune) {
	next := rune(0x100)
	for b := range alphabet {
		if '!' <= b && b <= '~' || 0xA1 <= b && b <= 0xAC || 0xAE <= b {
			alphabet[b] = rune(b)
		} else {
			alphabet[b] = next
			next++
		}
	}
	return alphabet
}()

// bpe is the byte-level BPE model of tokenizer.json.
type bpe struct {
	// byteIDs[b] is the id of the token that stands for the single byte b, or
	// -1 when the vocabulary has none.
	byteIDs [256]int32
	// merges says which adjacent tokens join, and into what.
	merges map[tokenPair]merge
}

// tokenPair is two adjacent token ids.
type tokenPair struct{ left, right int32 }

// merge is what a tokenPair joins into.
type merge struct {
	rank int // position in the file's merge list; lower ranks apply first
	id   int32
}

// newBPE builds the model from the base vocabulary and the merge list, each
// merge naming its left and right token. The two tokens and the token they
// join into must all be in the vocabulary.
func newBPE(vocab map[string]int32, merges [][2]string) (*bpe, error) {
	model := &bpe{merges: make(map[tokenPair]merge, len(merges))}
	for b, r := range byteRunes {
		id, ok := vocab[string(r)]
		if !ok {
			id = -1
		}
		model.byteIDs[b] = id
	}
	for rank, parts := range merges {
		var ids [3]int32
		for i, token := range [3]string{parts[0], parts[1], parts[0] + parts[1]} {
			id, ok := vocab[token]
			if !ok {
				return nil, fmt.Errorf("merges[%d] joins %q and %q, but %q is not in the vocabulary", rank, parts[0], parts[1], token)
			}
			ids[i] = id
		}
		model.merges[tokenPair{ids[0], ids[1]}] = merge{rank: rank, id: ids[2]}
	}
	return model, nil
}

// symbol is one token of a piece that is being merged. The symbols of a piece
// form a doubly linked list in text order, so a merge unlinks its right half
// without moving the rest.
type symbol struct {
	id         int32
	prev, next int  // indices of the neighbours, -1 at either end
	absorbed   bool // merged into its left neighbour, no longer in the list
}

// candidate is a merge that applied, when it was queued, to the symbol at pos
// and its right neighbour.
type candidate struct {
	pos int
	merge
}

// before orders candidates the way the reference applies them: lowest rank
// first, and among equal ranks the leftmost first.
func (c candidate) before(other candidate) bool {
	if c.rank != other.rank {
		return c.rank < other.rank
	}
	return c.pos < other.pos
}

// candidateQueue is a binary min-heap of candidates: its first element is
// the one that goes before all others. It is written out here because
// container/heap would allocate for every candidate pushed.
type candidateQueue []candidate

// push adds c to the queue.
func (q *candidateQueue) push(c candidate) {
	heap := append(*q, c)
	for i := len(heap) - 1; i > 0; {
		parent := (i - 1) / 2
		if !heap[i].before(heap[parent]) {
			break
		}
		heap[i], heap[parent] = heap[parent], heap[i]
		i = parent
	}
	*q = heap
}

// pop removes and returns the candidate that goes first. The queue must not
// be empty.
func (q *candidateQueue) pop() candidate {
	heap := *q
	top := heap[0]
	last := len(heap) - 1
	heap[0] = heap[last]
	heap = heap[:last]
	for i := 0; ; {
		first := i
		for child := 2*i + 1; child <= 2*i+2 && child < len(heap); child++ {
			if heap[child].before(heap[first]) {
				first = child
			}
		}
		if first == i {
			break
		}
		heap[i], heap[first] = heap[first], heap[i]
		i = first
	}
	*q = heap
	return top
}

// merger is the working state for running BPE on one piece at a time. One
// merger serves every piece of an Encode call so that its buffers are reused;
// it must not be shared between goroutines.
type merger struct {
	model   *bpe
	symbols []symbol
	queue   candidateQueue
}

// appendPiece appends the ids of one pre-tokenized piece to ids.
//
// The piece starts as one token per byte; like the reference without an
// unknown token, bytes the vocabulary lacks are dropped. Merges then apply in
// rank order, not left to right. A queue of candidates makes that take
// O(n log n) for a piece of n bytes, which matters for long runs of letters
// or spaces. Queued candidates are not removed when a merge elsewhere changes
// a symbol they refer to; they are recognised as stale when popped.
func (m *merger) appendPiece(ids []int32, piece []rune) []int32 {
	m.symbols = m.symbols[:0]
	var encoded [utf8.UTFMax]byte
	for _, r := range piece {
		for _, b := range encoded[:utf8.EncodeRune(encoded[:], r)] {
			if id := m.model.byteIDs[b]; id >= 0 {
				m.symbols = append(m.symbols, symbol{id: id, prev: len(m.symbols) - 1, next: len(m.symbols) + 1})
			}
		}
	}
	if len(m.symbols) == 0 {
		return ids
	}
	m.symbols[len(m.symbols)-1].next = -1

	for pos := range m.symbols {
		m.enqueue(pos)
	}
	for len(m.queue) > 0 {
		top := m.queue.pop()
		left := &m.symbols[top.pos]
		if left.absorbed || left.next < 0 {
			continue
		}
		right := &m.symbols[left.next]
		if current, ok := m.model.merges[tokenPair{left.id, right.id}]; !ok || current.id != top.id {
			continue
		}

		left.id = top.id
		left.next = right.next
		right.absorbed = true
		if left.next >= 0 {
			m.symbols[left.next].prev = top.pos
		}
		if left.prev >= 0 {
			m.enqueue(left.prev)
		}
		m.enqueue(top.pos)
	}

	// The first symbol is never absorbed, so the list starts there.
	for pos := 0; pos >= 0; pos = m.symbols[pos].next {
		ids = append(ids, m.symbols[pos].id)
	}
	return ids
}

// enqueue queues the merge of the symbol at pos with its right neighbour, if
// the model has one for them.
func (m *merger) enqueue(pos int) {
	left := m.symbols[pos]
	if left.next < 0 {
		return
	}
	if found, ok := m.model.merges[tokenPair{left.id, m.symbols[left.next].id}]; ok {
		m.queue.push(candidate{pos, found})
	}
}
