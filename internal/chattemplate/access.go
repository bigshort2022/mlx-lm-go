package chattemplate

import (
	"errors"
	"fmt"
	"math"
	"math/big"
	"slices"
	"strings"
)

// Attribute, item and slice access, following jinja2's Environment.getattr
// and getitem: a missing attribute or item is undefined rather than an error,
// and each of the two falls back to the other.

// dictMethods are the attributes Python resolves to a method of dict before
// looking at the dict's keys. Chat templates read keys, so rather than hand a
// template a method where it expects a value, access to these names fails.
var dictMethods = map[string]bool{
	"clear": true, "copy": true, "fromkeys": true, "get": true, "items": true, "keys": true,
	"pop": true, "popitem": true, "setdefault": true, "update": true, "values": true,
}

// getAttr is object.name.
func getAttr(object any, name string) (any, error) {
	switch object := object.(type) {
	case undefined:
		return nil, object.err()
	case *dict:
		if dictMethods[name] {
			return nil, fmt.Errorf("dict.%s is a method in Python and is not supported; write [%q] to read the key", name, name)
		}
		if value, ok := object.values[name]; ok {
			return value, nil
		}
	case *namespace:
		if value, ok := object.attrs[name]; ok {
			return value, nil
		}
	case *loopState:
		if object.filter != nil {
			if value, ok, err := object.filteredAttr(name); ok || err != nil {
				return value, err
			}
		} else if value, ok := object.attr(name); ok {
			return value, nil
		}
	case string:
		if method, ok := stringMethods[name]; ok {
			return &function{func(args []any, kwargs []namedValue) (any, error) {
				return method(object, args, kwargs)
			}}, nil
		}
	case safeString:
		return nil, errors.New("attribute access on a safe string is not supported")
	case *macro:
		return nil, errors.New("attribute access on a macro is not supported")
	}
	if kind := typeName(object); slices.Contains(pythonAttributes[kind], name) {
		return nil, fmt.Errorf("%s.%s is not supported", kind, name)
	}
	return undefined{hint: fmt.Sprintf("'%s' has no attribute '%s'", objectDescription(object), name)}, nil
}

// objectDescription names a value the way Jinja's undefined messages do.
func objectDescription(object any) string {
	if object == nil {
		return "None"
	}
	return typeName(object) + " object"
}

// getItem is object[key].
func getItem(object, key any) (any, error) {
	switch object := object.(type) {
	case undefined:
		return nil, object.err()
	case safeString:
		return nil, fmt.Errorf("subscripting a safe string is not supported")
	case *dict:
		if name, ok := key.(string); ok {
			if value, ok := object.values[name]; ok {
				return value, nil
			}
		}
	case []any:
		if index, ok := sequenceIndex(key, len(object)); ok {
			return object[index], nil
		}
	case tuple:
		if index, ok := sequenceIndex(key, len(object)); ok {
			return object[index], nil
		}
	case string:
		if _, isInt := asIndex(key); isInt {
			chars := []rune(object)
			if index, ok := sequenceIndex(key, len(chars)); ok {
				return string(chars[index]), nil
			}
		}
	}
	if name, ok := key.(string); ok {
		return getAttr(object, name)
	}
	keyText, err := pyStr(key)
	if err != nil {
		keyText = typeName(key)
	}
	return undefined{hint: fmt.Sprintf("%s has no element %s", objectDescription(object), keyText)}, nil
}

// asIndex returns key as an integer if Python accepts it as a sequence index.
func asIndex(key any) (int, bool) {
	switch key := key.(type) {
	case int:
		return key, true
	case bool:
		if key {
			return 1, true
		}
		return 0, true
	}
	return 0, false
}

// sequenceIndex resolves key to a position in a sequence of the given length,
// counting from the end when negative.
func sequenceIndex(key any, length int) (int, bool) {
	index, ok := asIndex(key)
	if !ok {
		return 0, false
	}
	if index < 0 {
		index += length
	}
	return index, index >= 0 && index < length
}

// getSlice is object[start:stop:step], where an omitted part is nil. Unlike
// getItem, Jinja applies it without a safety net, so it raises what Python
// raises.
func getSlice(object, start, stop, step any) (any, error) {
	var length int
	switch object := object.(type) {
	case undefined:
		return nil, object.err()
	case safeString:
		return nil, fmt.Errorf("slicing a safe string is not supported")
	case []any:
		length = len(object)
	case tuple:
		length = len(object)
	case string:
		chars := []rune(object)
		indices, err := sliceIndices(len(chars), start, stop, step)
		if err != nil {
			return nil, err
		}
		var sliced strings.Builder
		for _, index := range indices {
			sliced.WriteRune(chars[index])
		}
		return sliced.String(), nil
	default:
		return nil, pyError("TypeError", "'%s' object is not subscriptable", typeName(object))
	}

	indices, err := sliceIndices(length, start, stop, step)
	if err != nil {
		return nil, err
	}
	sliced := make([]any, len(indices))
	if items, isTuple := object.(tuple); isTuple {
		for i, index := range indices {
			sliced[i] = items[index]
		}
		return tuple(sliced), nil
	}
	for i, index := range indices {
		sliced[i] = object.([]any)[index]
	}
	return sliced, nil
}

// sliceIndices lists the positions a Python slice selects from a sequence of
// the given length.
func sliceIndices(length int, start, stop, step any) ([]int, error) {
	stride := 1
	if step != nil {
		var ok bool
		if stride, ok = sliceInt(step); !ok {
			return nil, errSliceIndex
		}
		if stride == 0 {
			return nil, pyError("ValueError", "slice step cannot be zero")
		}
	}
	// The lowest and highest positions a bound can clamp to.
	lower, upper := 0, length
	if stride < 0 {
		lower, upper = -1, length-1
	}
	bound := func(value any, missing int) (int, error) {
		if value == nil {
			return missing, nil
		}
		index, ok := sliceInt(value)
		if !ok {
			return 0, errSliceIndex
		}
		if index < 0 {
			index += length
		}
		return min(max(index, lower), upper), nil
	}
	first, last := lower, upper
	if stride < 0 {
		first, last = upper, lower
	}
	from, err := bound(start, first)
	if err != nil {
		return nil, err
	}
	to, err := bound(stop, last)
	if err != nil {
		return nil, err
	}

	// The positions are counted rather than walked: stepping an index by a
	// stride near the limits of int would wrap around.
	count := 0
	switch {
	case stride > 0 && from < to:
		count = (to-from-1)/stride + 1
	case stride < 0 && from > to:
		magnitude := uint(-(stride + 1)) + 1 // -stride, which the smallest int does not have
		count = int(uint(from-to-1)/magnitude) + 1
	}
	indices := make([]int, count)
	for i := range indices {
		indices[i] = from + i*stride
	}
	return indices, nil
}

// sliceInt returns a slice start, stop or step as an int. An integer beyond
// 64 bits counts as the largest or smallest int: either lies outside every
// sequence, which is all a slice asks of it.
func sliceInt(value any) (int, bool) {
	if integer, ok := value.(*big.Int); ok {
		if integer.Sign() < 0 {
			return math.MinInt, true
		}
		return math.MaxInt, true
	}
	return asIndex(value)
}

var errSliceIndex = pyError("TypeError", "slice indices must be integers or None or have an __index__ method")
