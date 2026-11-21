package chattemplate

// A template value is one of:
//
//	nil         none
//	undefined   a missing variable, attribute or item
//	bool
//	int
//	float64
//	string
//	safeString  text the safe filter marked as not needing escaping
//	[]any       list
//	tuple       tuple, such as the pairs the items filter yields
//	*dict       mapping with string keys
//	*generator  what the items filter returns
//	*namespace  namespace(...) object
//	*loopState  the loop variable inside a for body
//	callable    macro, global function or bound string method
//
// A template cannot modify a value, except a namespace through
// {% set ns.attr = ... %}. Every value belongs to one Render call.

// undefined is Jinja's Undefined: it renders as nothing and is falsy, but
// reading an attribute or item of it, or computing with it, is an error.
type undefined struct {
	// hint says what was missing, in Jinja's words.
	hint string
}

// err is the error raised when the undefined value is used for more than a
// truth test.
func (u undefined) err() error {
	return pyError("UndefinedError", "%s", u.hint)
}

type tuple []any

type safeString string

// dict is a mapping that remembers the order its keys were added in, as
// Python's dict does.
type dict struct {
	keys   []string
	values map[string]any
}

func newDict(capacity int) *dict {
	return &dict{keys: make([]string, 0, capacity), values: make(map[string]any, capacity)}
}

// set adds a key or, as Python does, replaces the value of an existing key
// without moving it.
func (d *dict) set(key string, value any) {
	if _, exists := d.values[key]; !exists {
		d.keys = append(d.keys, key)
	}
	d.values[key] = value
}
