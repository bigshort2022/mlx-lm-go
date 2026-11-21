package chattemplate

import (
	"testing"
)

// fuzzSources are small templates that hand request values to every
// operation taking an index, a bound, a count or a size, one operation per
// template so that an error in one does not hide the next. The variables a
// to d come from the kwargs.
var fuzzSources = []string{
	"{{ a[b:c:d] }}",
	"{{ a[b] }}|{{ a[b][c] }}|{{ a.b }}|{{ a[b].c }}",
	"{{ messages[a:b:c] | tojson }}|{{ messages[a].content[b:c:d] }}",
	"{{ a + b }}",
	"{{ a - b }}|{{ -a }}",
	"{{ a < b }}|{{ a >= b }}",
	"{{ a == b }}|{{ a != c }}|{{ a in b }}|{{ a not in c }}",
	"{{ a ~ b ~ c ~ d }}|{{ a }}|{{ b | string }}|{{ c | safe }}|{{ d | trim }}",
	"{{ a | tojson(indent=b) }}",
	"{{ a | tojson(ensure_ascii=b) }}|{{ a | length }}",
	"{% for k, v in a | items %}{{ k }}={{ v | tojson }};{% endfor %}",
	"{{ a.split(b) }}|{{ a.split(b)[c] }}",
	"{{ a.strip(b) }}|{{ a.lstrip(c) }}|{{ a.rstrip(d) }}",
	"{{ a.startswith(b) }}|{{ a.endswith(c) }}",
	"{% for x, y in a %}{{ loop.index }}{{ loop.revindex0 }}{{ loop.previtem }}{{ loop.nextitem }}{{ x }}{{ y }}{% endfor %}",
	"{% for x in a %}{{ a[loop.index0 - b] }}{% for y in x %}{{ y }}{{ loop.depth }}{% endfor %}{% endfor %}",
	"{% macro m(x, y=b) %}{{ x[y] }}{% if x %}{{ m(x[1:], y) }}{% endif %}{% endmacro %}{{ m(a) }}{{ m(a, c) }}{{ m(y=a, x=d) }}",
	"{% set ns = namespace(v=a) %}{% for x in b %}{% set ns.v = ns.v + x %}{% endfor %}{{ ns.v }}",
	"{{ a is string }}{{ a is mapping }}{{ a is iterable }}{{ a is sequence }}{{ a is none }}{{ a is false }}{{ a is defined }}{% if a %}T{% endif %}",
	"{{ raise_exception(a) }}",
	"{{ bos_token ~ eos_token }}{{ documents }}{{ tools | tojson }}{{ a if b else c }}{{ a and b or not c }}",
}

// fuzzTemplates returns the templates FuzzRender picks from: the two shipped
// ones, then fuzzSources.
func fuzzTemplates(f *testing.F) []*Template {
	var templates []*Template
	for _, family := range goldenFamilies {
		tmpl, err := Load(vendoredDir(family.name))
		if err != nil {
			f.Fatal(err)
		}
		templates = append(templates, tmpl)
	}
	for _, source := range fuzzSources {
		tmpl, err := Parse(source, map[string]string{"eos_token": "</s>"})
		if err != nil {
			f.Fatalf("Parse(%q): %v", source, err)
		}
		templates = append(templates, tmpl)
	}
	return templates
}

// Whatever a request holds, Render returns: a prompt or an error, never a
// panic. The seeds are every fixture request against both shipped templates
// and values at the edges of int against the small ones.
func FuzzRender(f *testing.F) {
	templates := fuzzTemplates(f)

	for family := range goldenFamilies {
		name := goldenFamilies[family].name
		rows := append(
			readRows[requestRow](f, goldenPath(name), goldenRowCount),
			readRows[requestRow](f, differentialPath(name), differentialRowCount)...)
		for _, row := range rows {
			f.Add(uint8(family), []byte(row.Messages), []byte(row.Tools), []byte(row.Kwargs), row.AddGenerationPrompt)
		}
	}
	edges := []string{
		`{"a": [1, 2, 3], "b": 1, "c": null, "d": 9223372036854775807}`,
		`{"a": "a,b,c", "b": null, "c": 1, "d": -9223372036854775808}`,
		`{"a": [[1, 2], "xy", [3, [4]]], "b": -9223372036854775808, "c": 9223372036854775807, "d": -1}`,
		`{"a": {"b": {"c": [1.5, 1e309, -1e309, 18446744073709551616]}, "k": "v"}, "b": "b", "c": "c", "d": 0}`,
		`{"a": 9223372036854775807, "b": 9223372036854775807, "c": -9223372036854775808, "d": 1e308}`,
		`{"a": 18446744073709551616, "b": [18446744073709551616, 1e309, null, true, "é"], "c": 1.5, "d": 2}`,
		`{"a": "  \u00a0text\u2028 ", "b": ",", "c": "", "d": " \u00a0"}`,
		`{"a": [], "b": [], "c": {}, "d": ""}`,
		`{"a": true, "b": false, "c": null, "d": 0.0}`,
	}
	for index := range fuzzSources {
		for _, kwargs := range edges {
			f.Add(uint8(len(goldenFamilies)+index), []byte(oneUserMessage), []byte(nil), []byte(kwargs), true)
		}
	}

	f.Fuzz(func(_ *testing.T, template uint8, messages, tools, kwargs []byte, addGenerationPrompt bool) {
		tmpl := templates[int(template)%len(templates)]
		// The outcome is Python's business, checked by the fixture tests;
		// here only returning matters.
		_, _ = tmpl.Render(Input{
			Messages:            messages,
			Tools:               tools,
			Kwargs:              kwargs,
			AddGenerationPrompt: addGenerationPrompt,
		})
	})
}
