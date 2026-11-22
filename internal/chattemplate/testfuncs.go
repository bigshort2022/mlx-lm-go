package chattemplate

// testFunc implements "value is test".
type testFunc func(value any) bool

// tests are the tests a template may use, as in jinja2/tests.py. They follow
// Python's duck typing, which is why a mapping and undefined are sequences.
var tests = map[string]testFunc{
	"defined": func(value any) bool {
		_, missing := value.(undefined)
		return !missing
	},
	"undefined": func(value any) bool {
		_, missing := value.(undefined)
		return missing
	},
	"none":  func(value any) bool { return value == nil },
	"true":  func(value any) bool { return value == true },
	"false": func(value any) bool { return value == false },
	"string": func(value any) bool {
		_, ok := asText(value)
		return ok
	},
	"mapping": func(value any) bool {
		_, ok := value.(*dict)
		return ok
	},
	// iterable: iter(value) works.
	"iterable": func(value any) bool {
		switch value.(type) {
		case string, safeString, []any, tuple, *dict, undefined, *generator, *loopState:
			return true
		}
		return false
	},
	// sequence: the value has a length and can be subscripted.
	"sequence": func(value any) bool {
		switch value.(type) {
		case string, safeString, []any, tuple, *dict, undefined:
			return true
		}
		return false
	},
}
