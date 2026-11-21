package chattemplate

// iterate lists what a for loop over the value visits: the items of a list
// or tuple, the keys of a mapping, the characters of a string, and nothing
// for undefined.
func iterate(value any) ([]any, error) {
	switch value := value.(type) {
	case []any:
		return value, nil
	case tuple:
		return value, nil
	case undefined:
		return nil, nil
	case *generator:
		return value.items, value.err
	case *dict:
		keys := make([]any, len(value.keys))
		for i, key := range value.keys {
			keys[i] = key
		}
		return keys, nil
	case string:
		chars := make([]any, 0, len(value))
		for _, char := range value {
			chars = append(chars, string(char))
		}
		return chars, nil
	}
	return nil, pyError("TypeError", "'%s' object is not iterable", typeName(value))
}

// unpack splits one loop item over count loop variables.
func unpack(item any, count int) ([]any, error) {
	values, err := iterate(item)
	switch {
	case err != nil:
		return nil, pyError("TypeError", "cannot unpack non-iterable %s object", typeName(item))
	case len(values) > count:
		return nil, pyError("ValueError", "too many values to unpack (expected %d)", count)
	case len(values) < count:
		return nil, pyError("ValueError", "not enough values to unpack (expected %d, got %d)", count, len(values))
	}
	return values, nil
}
