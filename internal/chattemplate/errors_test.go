package chattemplate

import (
	"errors"
	"testing"
)

func requireTemplateError(t *testing.T, source, want string) {
	t.Helper()
	got, err := renderSource(t, source, "")
	var templateErr *TemplateError
	if !errors.As(err, &templateErr) {
		t.Fatalf("Render returned (%q, %v), want *TemplateError", got, err)
	}
	if templateErr.Msg != want {
		t.Fatalf("TemplateError.Msg = %q, want %q", templateErr.Msg, want)
	}
	if got != "" {
		t.Fatalf("Render returned %q along with the error", got)
	}
}

func TestRaiseExceptionReturnsTemplateError(t *testing.T) {
	requireTemplateError(t, "before {{ raise_exception('boom') }} after", "boom")
}

func TestRaiseExceptionInsideBlocksReturnsTemplateError(t *testing.T) {
	requireTemplateError(t,
		"{% for message in messages %}{% if message.role == 'user' %}{{ raise_exception('no ' + message.role) }}{% endif %}{% endfor %}",
		"no user")
}

// The Qwen3.5 template raises from a macro whose result is piped into trim.
func TestRaiseExceptionInsideFilteredMacroCallReturnsTemplateError(t *testing.T) {
	requireTemplateError(t,
		"{% macro check(x) %}{{ raise_exception('bad ' + x) }}{% endmacro %}{% set out = check('input')|trim %}{{ out }}",
		"bad input")
}

func TestParseRejectsMalformedTemplate(t *testing.T) {
	if _, err := Parse("{% if %}", nil); err == nil {
		t.Fatal("Parse accepted a malformed template")
	}
}
