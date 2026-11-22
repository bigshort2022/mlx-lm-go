package chattemplate

import (
	"errors"
	"testing"
)

func TestFilteredItemsLaziness(t *testing.T) {
	calls := 0
	skip := false
	c := filteredItems{source: []any{0, 1, nil, 3}, accept: func(v any) (any, bool, error) {
		calls++
		return v, v != 0 && !skip, nil
	}}
	if v, ok, err := c.at(0); v != 1 || !ok || err != nil || calls != 2 {
		t.Fatalf("at(0) = (%v,%v,%v), calls=%d; want (1,true,nil), calls=2", v, ok, err, calls)
	}
	if v, ok, err := c.at(1); v != nil || !ok || err != nil || calls != 3 {
		t.Fatalf("at(1) = (%v,%v,%v), calls=%d; want (nil,true,nil), calls=3", v, ok, err, calls)
	}
	skip = true
	if n, err := c.length(); n != 2 || err != nil || calls != 4 {
		t.Fatalf("length = (%d,%v), calls=%d; want (2,nil), calls=4", n, err, calls)
	}
	if _, ok, err := c.at(2); ok || err != nil || calls != 4 {
		t.Fatalf("after end: ok=%v error=%v calls=%d", ok, err, calls)
	}
	if v, ok, err := c.at(0); v != 1 || !ok || err != nil || calls != 4 {
		t.Fatalf("memoized at(0) = (%v,%v,%v), calls=%d", v, ok, err, calls)
	}
}

func TestFilteredItemsStickyError(t *testing.T) {
	bad := errors.New("predicate failed")
	calls := 0
	c := filteredItems{source: []any{1, 2, 3}, accept: func(v any) (any, bool, error) {
		calls++
		if v == 2 {
			return nil, false, bad
		}
		return v, true, nil
	}}
	if _, _, err := c.at(9); !errors.Is(err, bad) || calls != 2 {
		t.Fatalf("at(9) error=%v calls=%d", err, calls)
	}
	if _, err := c.length(); !errors.Is(err, bad) || calls != 2 {
		t.Fatalf("length error=%v calls=%d", err, calls)
	}
	if v, ok, err := c.at(0); v != 1 || !ok || err != nil || calls != 2 {
		t.Fatalf("memoized result (%v,%v,%v), calls=%d", v, ok, err, calls)
	}
}
