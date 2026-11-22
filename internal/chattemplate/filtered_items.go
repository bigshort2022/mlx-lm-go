package chattemplate

// filteredItems lazily filters source items, memoizing the normalized values
// returned by accept. Each source item is tested at most once.
type filteredItems struct {
	source   []any
	accept   func(any) (any, bool, error)
	cursor   int
	accepted []any
	err      error
}

// at advances source until the zero-based accepted position i, returning
// the memoized value. ok is false on exhaustion.
func (f *filteredItems) at(i int) (any, bool, error) {
	if i < len(f.accepted) {
		return f.accepted[i], true, nil
	}
	if f.err != nil {
		return nil, false, f.err
	}
	for len(f.accepted) <= i {
		if f.cursor >= len(f.source) {
			return nil, false, nil
		}
		item := f.source[f.cursor]
		f.cursor++
		value, ok, err := f.accept(item)
		if err != nil {
			f.err = err
			return nil, false, err
		}
		if ok {
			f.accepted = append(f.accepted, value)
		}
	}
	return f.accepted[i], true, nil
}

// length drains the source once and returns the accepted count.
func (f *filteredItems) length() (int, error) {
	if _, _, err := f.at(len(f.source)); err != nil {
		return 0, err
	}
	return len(f.accepted), nil
}
