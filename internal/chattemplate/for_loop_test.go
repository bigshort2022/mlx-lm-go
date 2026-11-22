package chattemplate

import (
	"encoding/json"
	"testing"
)

func TestJinjaForLoopParity(t *testing.T) {
	const metadata = `{% for x in xs if x %}{{ loop.index }}/{{ loop.index0 }}/{{ loop.length }}/{{ loop.revindex }}/{{ loop.revindex0 }}/{{ loop.first }}/{{ loop.last }}/{{ loop.previtem|default("-") }}/{{ x }}/{{ loop.nextitem|default("-") }};{% else %}EMPTY{% endfor %}`
	cases := []struct{ name, source, kwargs, want string }{
		{"filtered metadata", metadata, `{"xs":[0,2,0,4,0]}`, "1/0/2/2/1/True/False/-/2/4;2/1/2/1/0/False/True/2/4/-;"},
		{"filtered empty", metadata, `{"xs":[0,false,null]}`, "EMPTY"},
		{"ordinary empty", `{% for x in xs %}{{ x }}{% else %}EMPTY{% endfor %}`, `{"xs":[]}`, "EMPTY"},
		{"ordinary nonempty", `{% for x in xs %}{% else %}EMPTY{% endfor %}`, `{"xs":[0]}`, ""},
		{"unpack generator", `{% for name, ok in m|items if ok %}{{ name }};{% else %}EMPTY{% endfor %}`, `{"m":{"skip":false,"pass":true,"last":true}}`, "pass;last;"},
		{"else scope", `{% set flag="outer" %}{% for x in xs if x %}{{ x }}{% else %}{% set flag="inner" %}{{ x|default("missing") }}-{{ flag }}{% endfor %}|{{ flag }}`, `{"xs":[0],"x":"parent"}`, "parent-inner|outer"},
		{"outer loop visible", `{% for x in xs %}{% for y in ys if loop.index == 1 %}{{ y }}{% else %}{{ loop.index }}{% endfor %};{% endfor %}`, `{"xs":[1,2],"ys":["A","B"]}`, "AB;2;"},
		{"lazy predicate", `{% set ns=namespace(skip=false) %}{% for x in xs if not ns.skip %}{{ x }}{% set ns.skip=true %}{% else %}EMPTY{% endfor %}`, `{"xs":[1,2,3]}`, "1"},
		{"metadata lookahead", `{% set ns=namespace(skip=false) %}{% for x in xs if not ns.skip %}{{ x }}:{{ loop.nextitem|default("-") }};{% set ns.skip=true %}{% endfor %}`, `{"xs":[1,2,3]}`, "1:2;2:-;"},
		{"length drains remainder", `{% set ns=namespace(skip=false) %}{% for x in xs if not ns.skip %}{{ x }}:{{ loop.length }};{% set ns.skip=true %}{% endfor %}`, `{"xs":[1,2,3]}`, "1:3;2:3;3:3;"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			tmpl, err := Parse(c.source, nil)
			if err != nil {
				t.Fatal(err)
			}
			got, err := tmpl.Render(Input{Messages: oneUserMessage, Kwargs: json.RawMessage(c.kwargs)})
			if err != nil || got != c.want {
				t.Fatalf("got (%q, %v), want %q", got, err, c.want)
			}
		})
	}
}

func TestJinjaFilteredDestructuring(t *testing.T) {
	cases := []struct{ name, source, input, want string }{
		{"metadata returns a tuple", `{% for a,b in xs if a %}{{ loop.previtem|default("-") }};{% endfor %}`, `{"xs":[["x","y"],["z","w"]]}`, "-;('x', 'y');"},
		{"unpack one shot input once", `{% set g = m|items %}{% for a,b in (g,) if a %}{{ a }},{{ b }}{% endfor %}`, `{"m":{"a":1,"b":2}}`, "('a', 1),('b', 2)"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			tmpl, err := Parse(c.source, nil)
			if err != nil {
				t.Fatal(err)
			}
			got, err := tmpl.Render(Input{Messages: oneUserMessage, Kwargs: json.RawMessage(c.input)})
			if err != nil || got != c.want {
				t.Fatalf("got (%q, %v), want %q", got, err, c.want)
			}
		})
	}
}
