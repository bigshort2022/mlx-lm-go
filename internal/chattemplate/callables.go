package chattemplate

import (
	"fmt"
	"strings"
)

// callable is a value that can be called: a macro, a global function or a
// bound string method.
type callable interface {
	call(args []any, kwargs []namedValue) (any, error)
}

// function is a callable implemented in Go. It is a struct, not a bare func,
// so that two function values can be compared with ==.
type function struct {
	fn func(args []any, kwargs []namedValue) (any, error)
}

func (f *function) call(args []any, kwargs []namedValue) (any, error) {
	return f.fn(args, kwargs)
}

// unsupportedGlobals are the functions Jinja and Hugging Face give templates
// that this package does not implement.
var unsupportedGlobals = []string{"range", "dict", "lipsum", "cycler", "joiner", "strftime_now"}

// globals are the functions every template can call. A template variable of
// the same name hides them.
var globals = func() map[string]any {
	functions := map[string]any{
		"namespace":       &function{newNamespace},
		"raise_exception": &function{raiseException},
	}
	// The unsupported ones exist, so that "is defined" answers as in Python,
	// but cannot be called.
	for _, name := range unsupportedGlobals {
		functions[name] = &function{func([]any, []namedValue) (any, error) {
			return nil, fmt.Errorf("%s() is not supported", name)
		}}
	}
	return functions
}()

// namespace is the object namespace(...) returns: the one value a template
// can assign into, which is how a for body passes results out.
type namespace struct {
	attrs map[string]any
}

func newNamespace(args []any, kwargs []namedValue) (any, error) {
	if len(args) > 0 {
		return nil, pyError("TypeError", "namespace() with positional arguments is not supported")
	}
	ns := &namespace{attrs: make(map[string]any, len(kwargs))}
	for _, attr := range kwargs {
		ns.attrs[attr.name] = attr.value
	}
	return ns, nil
}

// raiseException is the raise_exception(message) Hugging Face gives chat
// templates to reject a conversation.
func raiseException(args []any, kwargs []namedValue) (any, error) {
	var message any
	switch {
	case len(args) == 1 && len(kwargs) == 0:
		message = args[0]
	case len(args) == 0 && len(kwargs) == 1 && kwargs[0].name == "message":
		message = kwargs[0].value
	default:
		return nil, pyError("TypeError", "raise_exception() takes exactly one argument, the message")
	}
	text, err := pyStr(message)
	if err != nil {
		return nil, err
	}
	return nil, &TemplateError{Msg: text}
}

// macro is a {% macro %} defined by the running template.
type macro struct {
	name   string
	params []macroParam
	body   []statement
	// root is the template's top-level scope: a macro body sees the
	// template's variables as they are when it is called, not the variables
	// of the for body it may be called from.
	root *scope
	// depth counts the calls of this macro in progress. A macro value belongs
	// to one render, so the count is not shared.
	depth int
}

// maxMacroDepth bounds how deep a macro may call itself. Python gives up a
// little under 200 levels; without a bound, a template that recurses forever
// would overflow the stack and take the process down.
const maxMacroDepth = 150

// macroParam is one parameter; fallback is nil when it has no default.
type macroParam struct {
	name     string
	fallback expr
}

func (m *macro) call(args []any, kwargs []namedValue) (any, error) {
	if m.depth == maxMacroDepth {
		return nil, pyError("RecursionError", "maximum recursion depth exceeded")
	}
	m.depth++
	defer func() { m.depth-- }()

	if len(args) > len(m.params) {
		return nil, pyError("TypeError", "macro '%s' takes not more than %d argument(s)", m.name, len(m.params))
	}
	for _, keyword := range kwargs {
		// A keyword must name a parameter that was not given by position.
		if m.paramIndex(keyword.name) < len(args) {
			return nil, pyError("TypeError", "macro '%s' takes no keyword argument '%s'", m.name, keyword.name)
		}
	}

	sc := m.root.child()
	for i, param := range m.params {
		var value any
		keyword := findNamed(kwargs, param.name)
		switch {
		case i < len(args):
			value = args[i]
		case keyword != nil:
			value = keyword.value
		case param.fallback != nil:
			var err error
			if value, err = param.fallback.eval(sc); err != nil {
				return nil, err
			}
		default:
			value = undefined{hint: "parameter '" + param.name + "' was not provided"}
		}
		sc.set(param.name, value)
	}

	var out strings.Builder
	if err := execAll(m.body, &out, sc); err != nil {
		return nil, err
	}
	return out.String(), nil
}

// paramIndex returns the position of the named parameter, or -1.
func (m *macro) paramIndex(name string) int {
	for i, param := range m.params {
		if param.name == name {
			return i
		}
	}
	return -1
}

func findNamed(named []namedValue, name string) *namedValue {
	for i := range named {
		if named[i].name == name {
			return &named[i]
		}
	}
	return nil
}
