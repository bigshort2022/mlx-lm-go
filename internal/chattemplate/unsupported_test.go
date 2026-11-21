package chattemplate

import (
	"strings"
	"testing"
)

// Jinja has far more constructs than chat templates use. One this package
// does not implement must stop Parse with an error that names it: rendering
// around it would produce a prompt Python does not.
func TestParseNamesTheUnsupportedConstruct(t *testing.T) {
	cases := []struct {
		source string
		want   string // must appear in the error
	}{
		// Statements.
		{"{% raw %}{{ x }}{% endraw %}", `statement "raw" is not supported`},
		{"{% include 'other.jinja' %}", `statement "include" is not supported`},
		{"{% for x in l %}{% break %}{% endfor %}", `statement "break" is not supported`},
		{"{% generation %}x{% endgeneration %}", `statement "generation" is not supported`},
		{"{% call m() %}x{% endcall %}", `statement "call" is not supported`},
		{"{% for x in l %}a{% else %}b{% endfor %}", "for ... else is not supported"},
		{"{% for x in l if x %}a{% endfor %}", "for ... if is not supported"},
		{"{% for x in l recursive %}a{% endfor %}", "for ... recursive is not supported"},
		{"{% set x %}a{% endset %}", "block set"},
		{"{% set a, b = l %}", "set with several targets is not supported"},
		// Operators.
		{"{{ 2 * 3 }}", `operator "*" is not supported`},
		{"{{ 6 / 3 }}", `operator "/" is not supported`},
		{"{{ 7 // 2 }}", `operator "//" is not supported`},
		{"{{ 7 % 2 }}", `operator "%" is not supported`},
		{"{{ 2 ** 3 }}", `operator "**" is not supported`},
		{"{{ +3 }}", `unary "+" is not supported`},
		{"{{ 1 < 2 < 3 }}", "chained comparisons are not supported"},
		// Literals.
		{"{{ [1, 2] }}", "list literals are not supported"},
		{"{{ {'a': 1} }}", "dict literals are not supported"},
		{"{{ (1, 2) }}", "tuple literals are not supported"},
		{"{{ 1.5 }}", "float literals are not supported"},
		{"{{ 0x1f }}", `unsupported number literal "0x"`},
		{"{{ 1_000 }}", `unsupported number literal "1_"`},
		{`{{ '\x41' }}`, `unsupported string escape \x`},
		{"{{ 'a' 'b' }}", `unexpected string "b"`},
		// Filters and tests.
		{"{{ s | upper }}", `filter "upper" is not supported`},
		{"{{ q | default('x') }}", `filter "default" is not supported`},
		{"{{ l | join(', ') }}", `filter "join" is not supported`},
		{"{{ s | trim('x') }}", `filter "trim": positional arguments are not supported`},
		{"{{ v | tojson(2) }}", `filter "tojson": positional arguments are not supported`},
		{"{{ v | tojson(sort_keys=true) }}", `filter "tojson": argument "sort_keys" is not supported`},
		{"{{ v | tojson(separators=none) }}", `filter "tojson": argument "separators" is not supported`},
		{"{{ i is number }}", `test "number" is not supported`},
		{"{{ i is divisibleby 3 }}", `test "divisibleby" is not supported`},
		{"{{ i is defined(1) }}", `test "defined": arguments are not supported`},
		{"{{ t is true is true }}", `test "true": arguments are not supported`},
		// Calls.
		{"{{ f(*l) }}", `argument unpacking with "*" is not supported`},
		{"{{ f(**d) }}", `argument unpacking with "**" is not supported`},
		{"{% for i in range(3) %}{{ i }}{% endfor %}", `function "range" is not supported`},
		{"{{ strftime_now('%Y-%m-%d') }}", `function "strftime_now" is not supported`},
		{"{{ dict(a=1) }}", `function "dict" is not supported`},
		{"{{ joiner(', ') }}", `function "joiner" is not supported`},
		{"{{ cycler('a', 'b') }}", `function "cycler" is not supported`},
		{"{{ lipsum(1) }}", `function "lipsum" is not supported`},
		{"{% for x in l %}{{ loop.cycle('a', 'b') }}{% endfor %}", "loop.cycle is not supported"},
		{"{% for x in l %}{{ loop.changed(x) }}{% endfor %}", "loop.changed is not supported"},
		// Macros.
		{"{% macro m() %}{{ varargs }}{% endmacro %}", `"varargs" in a macro is not supported`},
		{"{% macro m() %}{{ kwargs | tojson }}{% endmacro %}", `"kwargs" in a macro is not supported`},
		{"{% macro m() %}{% if x %}{{ caller() }}{% endif %}{% endmacro %}", `"caller" in a macro is not supported`},
		{"{% macro m(a, caller=none) %}{% endmacro %}", `macro parameter "caller" is not supported`},
		{"{% for x in l %}{% macro m() %}{{ x }}{% endmacro %}{{ m() }}{% endfor %}", "macro inside a for loop is not supported"},
		{"{% macro outer(a) %}{% macro inner() %}{{ a }}{% endmacro %}{% endmacro %}", "macro inside a macro is not supported"},
		// Python accepts these; see the package comment.
		{"{{ 99999999999999999999 }}", "integer literal 99999999999999999999 is out of range"},
		{"text {# never closed", "missing end of comment tag"},
	}
	for _, c := range cases {
		_, err := Parse(c.source, nil)
		switch {
		case err == nil:
			t.Errorf("Parse(%q) succeeded, want an error with %q", c.source, c.want)
		case !strings.Contains(err.Error(), c.want):
			t.Errorf("Parse(%q) = %q, want an error with %q", c.source, err, c.want)
		}
	}
}

func TestParseRejectsMalformedTemplates(t *testing.T) {
	for _, source := range []string{
		"{% if %}",
		"{% if t %}never closed",
		"{% endif %}",
		"{% if t %}a{% else t %}b{% endif %}",
		"{% if t %}a{% endif t %}",
		"{% for in l %}{% endfor %}",
		"{% for x l %}{% endfor %}",
		"{% for x in l %}{% endif %}",
		"{% set = 1 %}",
		"{% set true = 1 %}",
		"{% macro m(a=1, b) %}{% endmacro %}",
		"{% macro m %}{% endmacro %}",
		"{% 3 %}",
		"{{ }}",
		"{{ a b }}",
		"{{ a.3 }}",
		"{{ l[1 2] }}",
		"{{ l[0:1:1:1] }}",
		"{{ f(a=1, 2) }}",
		"{{ (a }}",
		"{{ a) }}",
		"{{ l[0) }}",
		"{{ 'unterminated }}",
		"{{ never closed",
		"{{ a ? b }}",
	} {
		if _, err := Parse(source, nil); err == nil {
			t.Errorf("Parse(%q) succeeded, want an error", source)
		}
	}
}

func TestParseErrorsReportTheLine(t *testing.T) {
	_, err := Parse("line one\n{% if true %}\n  {{- 'a'\n   | upper }}\n{% endif %}", nil)
	if err == nil || !strings.HasPrefix(err.Error(), "line 4: ") {
		t.Fatalf("Parse error = %v, want one that starts with %q", err, "line 4: ")
	}
}

// Python resolves these at render time in ways no chat template means; they
// fail rather than render something else.
func TestRenderNamesTheUnsupportedOperation(t *testing.T) {
	cases := []struct {
		source string
		want   string
	}{
		// In Python d.items is the dict method, even when the dict has such a key.
		{"{{ d.items }}", "dict.items is a method in Python"},
		{"{{ d.get('k') }}", "dict.get is a method in Python"},
		{"{{ d['keys'] }}", "dict.keys is a method in Python"},
		{"{{ s.upper() }}", "str.upper is not supported"},
		{"{{ s.split() }}", "str.split() takes exactly one argument"},
		{"{{ namespace(d) }}", "namespace() with positional arguments is not supported"},
		{"{{ (s|safe).strip() }}", "attribute access on a safe string is not supported"},
		{"{{ (s|safe)[0] }}", "subscripting a safe string is not supported"},
		{"{{ (s|safe)[1:] }}", "slicing a safe string is not supported"},
		{"{{ (s|safe) + 'x' }}", "unsupported operand type(s) for +: 'Markup' and 'str'"},
		{"{{ 'x' + (s|safe) }}", `can only concatenate str (not "Markup") to str`},
		{"{% for c in s|safe %}{{ c }}{% endfor %}", "'Markup' object is not iterable"},
		{"{{ namespace(a=1) }}", "cannot render a Namespace as text"},
		{"{{ d|items }}", "cannot render a generator as text"},
		{"{{ 'k' in d|items }}", "argument of type 'generator' is not iterable"},
		{"{{ l < l }}", "'<' not supported between instances of 'list' and 'list'"},
		// Reached through another name, the call is only seen while rendering.
		{"{% set r = range %}{{ r(3) }}", "range() is not supported"},
		{"{% set now = strftime_now %}{{ now('%Y') }}", "strftime_now() is not supported"},
		// Python has these attributes; nothing a chat template needs.
		{"{{ s.upper }}", "str.upper is not supported"},
		{"{{ s['upper'] }}", "str.upper is not supported"},
		{"{{ i.real }}", "int.real is not supported"},
		{"{{ t.real }}", "bool.real is not supported"},
		{"{{ f.is_integer() }}", "float.is_integer is not supported"},
		{"{{ l.index(1) }}", "list.index is not supported"},
		{"{{ l['count'] }}", "list.count is not supported"},
		{"{% for pair in d|items %}{{ pair.count }}{% endfor %}", "tuple.count is not supported"},
		{"{% macro m(a) %}{% endmacro %}{{ m.name }}", "attribute access on a macro is not supported"},
		{"{% macro m(a) %}{% endmacro %}{{ m['arguments'] }}", "attribute access on a macro is not supported"},
		// Python integers have no upper bound.
		{"{{ big + 1 }}", "integer result is out of range"},
		{"{{ 0 - big - 2 }}", "integer result is out of range"},
		{"{{ -small }}", "integer result is out of range"},
		{"{{ huge + 1 }}", "arithmetic on an integer beyond 64 bits is not supported"},
		{"{{ 1 - huge }}", "arithmetic on an integer beyond 64 bits is not supported"},
		{"{{ -huge }}", "arithmetic on an integer beyond 64 bits is not supported"},
		{"{{ huge > 1 }}", "ordering an integer beyond 64 bits is not supported"},
		// Python would write the megabytes of spaces asked for.
		{"{{ l | tojson(indent=101) }}", "tojson: indent 101 is above the limit of 100"},
		{"{{ l | tojson(indent=big) }}", "is above the limit of 100"},
		{"{{ l | tojson(indent=huge) }}", "tojson: indent must be an integer or none"},
	}
	for _, c := range cases {
		got, err := renderSource(t, c.source, `{"d": {"k": 1}, "s": "text", "l": [1], "i": 3, "f": 1.5, "t": true,
			"big": 9223372036854775807, "small": -9223372036854775808, "huge": 18446744073709551615}`)
		switch {
		case err == nil:
			t.Errorf("%s rendered %q, want an error with %q", c.source, got, c.want)
		case !strings.Contains(err.Error(), c.want):
			t.Errorf("%s: error %q, want one with %q", c.source, err, c.want)
		}
	}
}
