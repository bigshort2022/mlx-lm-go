package webui

import (
	"io/fs"
	"os"
	"regexp"
	"slices"
	"strings"
	"testing"
)

var uiFiles = []string{"index.html", "chat.js", "chat.css"}

func TestEmbeddedUIHasItsThreeFiles(t *testing.T) {
	for _, name := range uiFiles {
		body, err := fs.ReadFile(FS(), name)
		if err != nil {
			t.Errorf("%s: %v", name, err)
		} else if len(body) == 0 {
			t.Errorf("%s is empty", name)
		}
	}
}

// The UI is served under a policy that only allows its own origin, and must
// work offline, so it may not load or call anything on another host. The
// loopback names may appear, as placeholders and examples, and so may a
// scheme in front of an address the page computes, such as `http://${text}`.
func TestEmbeddedUIReferencesNoOutsideHost(t *testing.T) {
	for _, name := range uiFiles {
		body, err := fs.ReadFile(FS(), name)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if refs := outsideHosts(string(body)); len(refs) != 0 {
			t.Errorf("%s refers to other hosts: %q", name, refs)
		}
	}
}

func TestOutsideHostsFindsReferences(t *testing.T) {
	cases := []struct {
		text string
		want []string // the references reported
	}{
		// Rejected.
		{`<link href="https://fonts.googleapis.com/x" rel="stylesheet">`, []string{"https://fonts.googleapis.com"}},
		{`<script src="//cdn.example/x.js"></script>`, []string{`src="//cdn.example`}},
		{`<a href='//cdn.example/x.js'>`, []string{`href='//cdn.example`}},
		{`<img src=//cdn.example/x.png>`, []string{`src=//cdn.example`}},
		{`fetch("http://api.example.com:8080/v1")`, []string{"http://api.example.com:8080"}},
		{`@import url(//cdn.example/x.css);`, []string{"url(//cdn.example"}},
		{`background: url("HTTPS://CDN.EXAMPLE/x.png")`, []string{"HTTPS://CDN.EXAMPLE"}},
		{`"http://127.0.0.1.evil.example/"`, []string{"http://127.0.0.1.evil.example"}},
		{`"http://localhost.evil.example/"`, []string{"http://localhost.evil.example"}},
		{`"http://127.0.0.1@evil.example/"`, []string{"http://127.0.0.1@evil.example"}},
		{`"http://[2001:db8::1]/"`, []string{"http://[2001:db8::1]"}},
		{`a "https://one.example" b 'https://two.example/x'`, []string{"https://one.example", "https://two.example"}},
		// Allowed.
		{`var EXAMPLE_SERVER = "http://127.0.0.1:8080";`, nil},
		{`placeholder="http://localhost:8080/v1"`, nil},
		{"new URL(ok ? text : `http://${text}`)", nil},
		{"`https://${host}/v1`", nil},
		{`<link rel="stylesheet" href="./chat.css" /><script src="/chat.js">`, nil},
		{`/^[a-z][a-z0-9+.-]*:\/\//i.test(text) // not a URL`, nil},
		{`url.protocol !== "http:" && url.protocol !== "https:"`, nil},
	}
	for _, tc := range cases {
		if got := outsideHosts(tc.text); !slices.Equal(got, tc.want) {
			t.Errorf("outsideHosts(%s) = %q, want %q", tc.text, got, tc.want)
		}
	}
}

func TestVersionMatchesVersionFile(t *testing.T) {
	recorded, err := os.ReadFile("VERSION")
	if err != nil {
		t.Fatal(err)
	}
	want := strings.TrimSpace(string(recorded))
	if want == "" {
		t.Fatal("VERSION is empty")
	}
	if got := Version(); got != want {
		t.Fatalf("Version() = %q, want %q from the VERSION file", got, want)
	}
}

var (
	// absoluteURL matches a scheme followed by a literal authority. An
	// authority the page computes, as in `http://${text}`, does not match.
	absoluteURL = regexp.MustCompile("(?i)https?://([a-z0-9\\[][^\\s/?#\"'<>`)\\\\]*)")
	// schemeRelative matches //authority where a page loads things from: src
	// and href attributes and CSS url().
	schemeRelative = regexp.MustCompile("(?i)(?:\\b(?:src|href)\\s*=\\s*|url\\(\\s*)[\"']?//([^\\s/?#\"'<>`)\\\\]+)")
)

// outsideHosts returns the references in text to hosts other than this
// machine's loopback names, in order of appearance within each kind.
func outsideHosts(text string) []string {
	var refs []string
	for _, pattern := range []*regexp.Regexp{absoluteURL, schemeRelative} {
		for _, match := range pattern.FindAllStringSubmatch(text, -1) {
			if host := hostOf(match[1]); host != "127.0.0.1" && host != "localhost" {
				refs = append(refs, match[0])
			}
		}
	}
	return refs
}

// hostOf returns the lower-case host of a URL authority, without user
// information and port.
func hostOf(authority string) string {
	host := strings.ToLower(authority[strings.LastIndex(authority, "@")+1:])
	if strings.HasPrefix(host, "[") {
		return host // an IPv6 literal; none is allowed
	}
	host, _, _ = strings.Cut(host, ":")
	return host
}
