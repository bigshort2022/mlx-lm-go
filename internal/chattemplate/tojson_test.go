package chattemplate

import "testing"

// The want strings are what Hugging Face's tojson filter, which is
// json.dumps(value, ensure_ascii=False), returned for json.loads(value).
func TestToJSONMatchesPythonDumps(t *testing.T) {
	cases := []struct {
		name  string
		value string
		want  string
	}{
		{"keys keep request order", `{"zeta":1,"alpha":{"m":true,"b":null,"a":[3,2,1]},"beta":"x"}`, `{"zeta": 1, "alpha": {"m": true, "b": null, "a": [3, 2, 1]}, "beta": "x"}`},
		{"html is not escaped", `"<b>bold</b> & \"quoted\" 'single'"`, `"<b>bold</b> & \"quoted\" 'single'"`},
		{"newline tab backslash", `"line one\nline two\ttab\\slash"`, `"line one\nline two\ttab\\slash"`},
		{"non-ascii is written as is", `"é 天气 😀"`, `"é 天气 😀"`},
		{"only C0 controls are escaped", `"\u0000\u001f\u007f\u2028"`, "\"\\u0000\\u001f\u007f\u2028\""},
		{"integer", `1`, `1`},
		{"negative integer", `-7`, `-7`},
		{"large integer", `123456789012345678`, `123456789012345678`},
		{"float", `1.5`, `1.5`},
		{"whole float keeps its fraction", `1.0`, `1.0`},
		{"negative zero", `-0.0`, `-0.0`},
		{"small exponent", `1e-9`, `1e-09`},
		{"exponent with fraction", `2.5e-7`, `2.5e-07`},
		{"smallest positional float", `0.0001`, `0.0001`},
		{"largest exponent float below one", `0.00001`, `1e-05`},
		{"exponent literal of a whole number", `1E5`, `100000.0`},
		{"large exponent", `1e16`, `1e+16`},
		{"true", `true`, `true`},
		{"false", `false`, `false`},
		{"null", `null`, `null`},
		{"empty object", `{}`, `{}`},
		{"empty array", `[]`, `[]`},
		{"nested empties", `[1,"a",{"k":[]}]`, `[1, "a", {"k": []}]`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := mustRenderSource(t, "{{ v | tojson }}", `{"v": `+c.value+`}`)
			if got != c.want {
				t.Fatalf("tojson(%s)\n got: %s\nwant: %s", c.value, got, c.want)
			}
		})
	}
}

func TestToJSONIndentMatchesPythonDumps(t *testing.T) {
	got := mustRenderSource(t, "{{ v | tojson(indent=2) }}", `{"v": {"zeta": 1, "alpha": [1, 2], "e": {}}}`)
	want := "{\n  \"zeta\": 1,\n  \"alpha\": [\n    1,\n    2\n  ],\n  \"e\": {}\n}"
	if got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestItemsFilterKeepsRequestKeyOrder(t *testing.T) {
	got := mustRenderSource(t, "{% for k, v in d|items %}{{ k }}={{ v }};{% endfor %}", `{"d": {"zeta": 1, "alpha": 2, "mid": 3}}`)
	if want := "zeta=1;alpha=2;mid=3;"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestIteratingAnObjectKeepsRequestKeyOrder(t *testing.T) {
	got := mustRenderSource(t, "{% for k in d %}{{ k }};{% endfor %}", `{"d": {"zeta": 1, "alpha": 2, "mid": 3}}`)
	if want := "zeta;alpha;mid;"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestToJSONEscapesCarriageReturnBackspaceAndFormFeed(t *testing.T) {
	got := mustRenderSource(t, "{{ v | tojson }}", `{"v": "a\rb\bc\fd"}`)
	if want := `"a\rb\bc\fd"`; got != want {
		t.Fatalf("got %s, want %s", got, want)
	}
}

func TestToJSONEnsureASCIIEscapesEverythingOutsidePrintableASCII(t *testing.T) {
	got := mustRenderSource(t, "{{ v | tojson(ensure_ascii=true) }}", "{\"v\": \"é 天气 😀 \u007f\"}")
	if want := `"\u00e9 \u5929\u6c14 \ud83d\ude00 \u007f"`; got != want {
		t.Fatalf("got %s, want %s", got, want)
	}
}

func TestToJSONEnsureASCIIFalseIsTheDefault(t *testing.T) {
	got := mustRenderSource(t, "{{ v | tojson(ensure_ascii=false) }}", `{"v": "é"}`)
	if want := `"é"`; got != want {
		t.Fatalf("got %s, want %s", got, want)
	}
}
