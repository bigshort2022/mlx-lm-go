package server

import (
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"testing"
	"testing/fstest"

	"github.com/bigshort2022/mlx-lm-go/internal/webui"
)

// testUI is a stand-in for the embedded chat UI.
var testUI = fstest.MapFS{
	"index.html":       {Data: []byte(`<!doctype html><script type="module" src="./chat.js"></script>`)},
	"chat.js":          {Data: []byte(`console.log("chat")`)},
	"chat.css":         {Data: []byte(`body { margin: 0 }`)},
	"assets/logo.svg":  {Data: []byte(`<svg xmlns="http://www.w3.org/2000/svg"/>`)},
	"assets/font.zzzz": {Data: []byte("\x00\x01 not text")},
}

// withUI makes the server serve testUI.
func withUI(h *harness) { h.ui = testUI }

func TestRootServesTheUIPage(t *testing.T) {
	got := newHarness(t, withUI).do("GET", "/", "", nil)
	if got.Code != 200 {
		t.Fatalf("status = %d, want 200: %s", got.Code, got.Body)
	}
	if ct := got.Header().Get("Content-Type"); ct != "text/html; charset=utf-8" {
		t.Errorf("Content-Type = %q, want text/html; charset=utf-8", ct)
	}
	if want := string(testUI["index.html"].Data); got.Body.String() != want {
		t.Errorf("body = %q, want %q", got.Body, want)
	}
}

func TestUIFilesAreServedWithTheirContentTypes(t *testing.T) {
	cases := map[string]string{
		"/chat.js":          "text/javascript; charset=utf-8",
		"/chat.css":         "text/css; charset=utf-8",
		"/index.html":       "text/html; charset=utf-8",
		"/assets/logo.svg":  "image/svg+xml",
		"/assets/font.zzzz": "application/octet-stream",
	}
	h := newHarness(t, withUI)
	for path, contentType := range cases {
		t.Run(path, func(t *testing.T) {
			got := h.do("GET", path, "", nil)
			if got.Code != 200 {
				t.Fatalf("status = %d, want 200: %s", got.Code, got.Body)
			}
			if ct := got.Header().Get("Content-Type"); ct != contentType {
				t.Errorf("Content-Type = %q, want %q", ct, contentType)
			}
			if want := string(testUI[path[1:]].Data); got.Body.String() != want {
				t.Errorf("body = %q, want %q", got.Body, want)
			}
		})
	}
}

func TestUIResponsesCarryTheSecurityHeaders(t *testing.T) {
	h := newHarness(t, withUI)
	for _, path := range []string{"/", "/index.html", "/chat.js", "/chat.css", "/assets/logo.svg"} {
		got := h.do("GET", path, "", nil)
		if got.Code != 200 {
			t.Fatalf("%s: status = %d, want 200", path, got.Code)
		}
		if csp := got.Header().Get("Content-Security-Policy"); csp != "default-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'" {
			t.Errorf("%s: Content-Security-Policy = %q", path, csp)
		}
		if cc := got.Header().Get("Cache-Control"); cc != "no-cache" {
			t.Errorf("%s: Cache-Control = %q, want no-cache", path, cc)
		}
		if v := got.Header().Get("X-Content-Type-Options"); v != "nosniff" {
			t.Errorf("%s: X-Content-Type-Options = %q, want nosniff", path, v)
		}
	}
}

func TestWithoutAUIRootIsNotFound(t *testing.T) {
	h := newHarness(t)
	for _, path := range []string{"/", "/index.html", "/chat.js"} {
		if got := h.do("GET", path, "", nil); got.Code != 404 || got.Body.String() != "Not Found" {
			t.Errorf("GET %s = %d %q, want 404 \"Not Found\"", path, got.Code, got.Body)
		}
	}
}

func TestUILeavesTheOtherRoutesAlone(t *testing.T) {
	h := newHarness(t, withUI, scripted(countPieces...))

	for _, path := range []string{"/nope", "/v1/nope", "/assets", "/assets/", "/../chat.js", "/assets/../chat.js", "//chat.js"} {
		if got := h.do("GET", path, "", nil); got.Code != 404 || got.Body.String() != "Not Found" {
			t.Errorf("GET %s = %d %q, want 404 \"Not Found\"", path, got.Code, got.Body)
		}
	}
	if got := h.replay(loadExchange(t, "health")); got.Code != 200 || !jsonEqual(t, got.Body.String(), `{"status": "ok"}`) {
		t.Errorf("GET /health = %d %s, want the health answer", got.Code, got.Body)
	}
	if got := h.replay(loadExchange(t, "models")); got.Code != 200 || got.Header().Get("Content-Type") != "application/json" {
		t.Errorf("GET /v1/models = %d %s, want the model list", got.Code, got.Body)
	}
	if got := h.replay(loadExchange(t, "chat")); got.Code != 200 || got.Header().Get("Content-Security-Policy") != "" {
		t.Errorf("POST /v1/chat/completions = %d with policy %q, want a plain completion", got.Code, got.Header().Get("Content-Security-Policy"))
	}
}

func TestUIRoutesCheckTheHostHeader(t *testing.T) {
	h := newHarness(t, withUI)
	for _, path := range []string{"/", "/chat.js", "/chat.css"} {
		got := h.do("GET", path, "", map[string]string{"Host": "evil.example:8080"})
		assertRejected(t, h, got, 403)
		if strings.Contains(got.Body.String(), "chat") {
			t.Errorf("GET %s under a foreign Host returned UI content: %s", path, got.Body)
		}
	}
}

func TestOnlyGetAndHeadReachTheUI(t *testing.T) {
	h := newHarness(t, withUI)
	for _, method := range []string{"POST", "PUT", "DELETE", "PATCH"} {
		for _, path := range []string{"/", "/chat.js"} {
			got := h.do(method, path, `{}`, nil)
			if got.Code != 404 && got.Code != 405 {
				t.Errorf("%s %s = %d, want 404 or 405", method, path, got.Code)
			}
			if strings.Contains(got.Body.String(), "chat") {
				t.Errorf("%s %s returned a UI file: %s", method, path, got.Body)
			}
		}
	}

	got := h.do("HEAD", "/chat.js", "", nil)
	if got.Code != 200 || got.Body.Len() != 0 {
		t.Fatalf("HEAD /chat.js = %d with body %q, want 200 and no body", got.Code, got.Body)
	}
	if ct := got.Header().Get("Content-Type"); ct != "text/javascript; charset=utf-8" {
		t.Errorf("HEAD Content-Type = %q", ct)
	}
	if n := got.Header().Get("Content-Length"); n != strconv.Itoa(len(testUI["chat.js"].Data)) {
		t.Errorf("HEAD Content-Length = %q, want the size of the file", n)
	}
	if got.Header().Get("Content-Security-Policy") == "" {
		t.Error("HEAD response has no Content-Security-Policy")
	}
}

// post sends a JSON POST with the given Origin to a live server.
func post(t *testing.T, url, origin, body string) *http.Response {
	t.Helper()
	req, err := http.NewRequest("POST", url, strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Origin", origin)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { resp.Body.Close() })
	return resp
}

// The page served at / calls the API on the same origin. That must work with
// no --allowed-origins at all, on whatever port the server listens.
func TestUIPageMayCallTheAPIWithoutAllowedOrigins(t *testing.T) {
	h, ts := newLiveHarness(t, withUI, scripted(countPieces...))
	if len(h.cfg.AllowedOrigins) != 0 {
		t.Fatalf("AllowedOrigins = %q, want none", h.cfg.AllowedOrigins)
	}
	own := fmt.Sprintf("http://127.0.0.1:%d", h.cfg.Port)
	if own != ts.URL {
		t.Fatalf("server listens at %s, config says %s", ts.URL, own)
	}

	resp := post(t, ts.URL+"/v1/chat/completions", own, chatRequest)
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != 200 {
		t.Fatalf("same-origin chat: status = %d, want 200: %s", resp.StatusCode, raw)
	}
	if content := decodeCompletion(t, string(raw)).Choices[0].Message.Content; content != "1, 2, 3, 4, 5" {
		t.Errorf("content = %q, want the completion", content)
	}
	if acao := resp.Header.Get("Access-Control-Allow-Origin"); acao != "" {
		t.Errorf("Access-Control-Allow-Origin = %q on a same-origin request, want none", acao)
	}

	// A page on another port of this machine is another origin.
	other := fmt.Sprintf("http://127.0.0.1:%d", h.cfg.Port+1)
	if resp := post(t, ts.URL+"/v1/chat/completions", other, chatRequest); resp.StatusCode != 403 {
		t.Errorf("chat from %s: status = %d, want 403", other, resp.StatusCode)
	}
	if n := h.engine.Calls.Load(); n != 1 {
		t.Errorf("engine ran %d times, want once", n)
	}
}

// The vendored build is servable: its page and the two files the page loads.
func TestVendoredUIIsServed(t *testing.T) {
	h := newHarness(t, func(h *harness) { h.ui = webui.FS() })
	page := h.do("GET", "/", "", nil)
	if page.Code != 200 || page.Header().Get("Content-Type") != "text/html; charset=utf-8" {
		t.Fatalf("GET / = %d %q", page.Code, page.Header().Get("Content-Type"))
	}
	for path, contentType := range map[string]string{
		"chat.js":  "text/javascript; charset=utf-8",
		"chat.css": "text/css; charset=utf-8",
	} {
		if !strings.Contains(page.Body.String(), `"./`+path+`"`) {
			t.Errorf("the page does not load ./%s", path)
		}
		got := h.do("GET", "/"+path, "", nil)
		if got.Code != 200 || got.Body.Len() == 0 || got.Header().Get("Content-Type") != contentType {
			t.Errorf("GET /%s = %d, %d bytes, %q", path, got.Code, got.Body.Len(), got.Header().Get("Content-Type"))
		}
	}
}
