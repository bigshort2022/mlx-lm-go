package server

import (
	"fmt"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
)

// chatRequest is a valid chat completion body.
const chatRequest = `{"messages": [{"role": "user", "content": "Count from 1 to 5."}]}`

// assertRejected checks that a request was refused with the given status and
// a JSON error, and that it never reached the engine.
func assertRejected(t *testing.T, h *harness, got *httptest.ResponseRecorder, status int) string {
	t.Helper()
	msg := errorMessage(t, got, status)
	if n := h.engine.Calls.Load(); n != 0 {
		t.Errorf("engine was called %d times", n)
	}
	return msg
}

func TestHostHeaderMustNameTheLoopbackServer(t *testing.T) {
	allowed := []string{"127.0.0.1:8080", "localhost:8080", "[::1]:8080", "LocalHost:8080"}
	for _, host := range allowed {
		t.Run("allowed "+host, func(t *testing.T) {
			h := newHarness(t, scripted(countPieces...))
			got := h.do("POST", "/v1/chat/completions", chatRequest, map[string]string{"Host": host})
			if got.Code != 200 {
				t.Fatalf("status = %d, want 200: %s", got.Code, got.Body)
			}
		})
	}

	// A page on another site whose DNS name was rebound to 127.0.0.1 still
	// sends its own name in Host.
	rejected := []string{"evil.example:8080", "evil.example", "127.0.0.1:9999", "127.0.0.1", "localhost", "localhost.evil.example:8080", "192.168.1.5:8080"}
	for _, host := range rejected {
		t.Run("rejected "+host, func(t *testing.T) {
			h := newHarness(t, scripted(countPieces...))
			got := h.do("POST", "/v1/chat/completions", chatRequest, map[string]string{"Host": host})
			if msg := assertRejected(t, h, got, 403); !strings.Contains(msg, host) {
				t.Errorf("error = %q, want it to name the host", msg)
			}
			if got := h.do("GET", "/health", "", map[string]string{"Host": host}); got.Code != 403 {
				t.Errorf("GET /health status = %d, want 403", got.Code)
			}
		})
	}
}

func TestHostHeaderOtherLoopbackConfigurations(t *testing.T) {
	cases := []struct {
		host   string // Config.Host
		port   int
		header string
		want   int
	}{
		{"localhost", 8080, "localhost:8080", 200},
		{"localhost", 8080, "evil.example:8080", 403},
		{"::1", 8080, "[::1]:8080", 200},
		{"::1", 8080, "evil.example:8080", 403},
		{"127.0.0.2", 8080, "evil.example:8080", 403},
		{"127.0.0.2", 8080, "127.0.0.2:8080", 200},
		// Browsers leave the default port out of Host.
		{"127.0.0.1", 80, "localhost", 200},
		{"127.0.0.1", 80, "127.0.0.1:80", 200},
		{"127.0.0.1", 80, "evil.example", 403},
	}
	for _, tc := range cases {
		t.Run(tc.host+" "+tc.header, func(t *testing.T) {
			h := newHarness(t, func(h *harness) { h.cfg.Host, h.cfg.Port = tc.host, tc.port })
			if got := h.do("GET", "/health", "", map[string]string{"Host": tc.header}); got.Code != tc.want {
				t.Fatalf("status = %d, want %d: %s", got.Code, tc.want, got.Body)
			}
		})
	}
}

func TestHostHeaderIsNotCheckedOffLoopback(t *testing.T) {
	for _, host := range []string{"0.0.0.0", "192.168.1.5", "mybox.local", ""} {
		t.Run(host, func(t *testing.T) {
			h := newHarness(t, scripted(countPieces...), func(h *harness) { h.cfg.Host = host })
			got := h.do("POST", "/v1/chat/completions", chatRequest, map[string]string{"Host": "anything.example:8080"})
			if got.Code != 200 {
				t.Fatalf("status = %d, want 200: %s", got.Code, got.Body)
			}
		})
	}
}

const chatPage = "https://unite4ai.com"

// allowOrigins configures --allowed-origins.
func allowOrigins(origins ...string) func(*harness) {
	return func(h *harness) { h.cfg.AllowedOrigins = origins }
}

// corsHeaders returns the CORS response headers that are set, by name.
func corsHeaders(got *httptest.ResponseRecorder) map[string]string {
	set := map[string]string{}
	for name, values := range got.Header() {
		if strings.HasPrefix(name, "Access-Control-") || name == "Vary" {
			set[name] = strings.Join(values, ", ")
		}
	}
	return set
}

func TestSameOriginRequestsAreAllowed(t *testing.T) {
	cases := []struct{ host, origin string }{
		{"127.0.0.1:8080", "http://127.0.0.1:8080"},
		{"localhost:8080", "http://localhost:8080"},
		{"[::1]:8080", "http://[::1]:8080"},
	}
	for _, tc := range cases {
		t.Run(tc.origin, func(t *testing.T) {
			h := newHarness(t, scripted(countPieces...))
			got := h.do("POST", "/v1/chat/completions", chatRequest, map[string]string{"Host": tc.host, "Origin": tc.origin})
			if got.Code != 200 {
				t.Fatalf("status = %d, want 200: %s", got.Code, got.Body)
			}
		})
	}
}

func TestListedOriginIsAllowed(t *testing.T) {
	rec := loadExchange(t, "chat_with_origin")
	h := newHarness(t, scripted(countPieces...), allowOrigins("https://other.example", chatPage))
	got := h.replay(rec)
	if got.Code != 200 {
		t.Fatalf("status = %d, want 200: %s", got.Code, got.Body)
	}
	if g, w := shapeOf(t, got.Body.String()), shapeOf(t, rec.Response.Body); !reflect.DeepEqual(g, w) {
		t.Fatalf("shape =\n%v, want\n%v", g, w)
	}
}

func TestWildcardAllowsAnyOrigin(t *testing.T) {
	h := newHarness(t, scripted(countPieces...), allowOrigins("*"))
	for _, origin := range []string{chatPage, "http://evil.example", "null"} {
		got := h.do("POST", "/v1/chat/completions", chatRequest, map[string]string{"Origin": origin})
		if got.Code != 200 {
			t.Fatalf("origin %s: status = %d, want 200: %s", origin, got.Code, got.Body)
		}
	}
}

func TestOtherOriginsAreForbidden(t *testing.T) {
	cases := []struct {
		name    string
		allowed []string
		host    string
		origin  string
	}{
		{"nothing allowed", nil, "127.0.0.1:8080", chatPage},
		{"not in the list", []string{"https://other.example"}, "127.0.0.1:8080", chatPage},
		{"lookalike of a listed origin", []string{chatPage}, "127.0.0.1:8080", chatPage + ".evil.example"},
		{"other scheme than listed", []string{chatPage}, "127.0.0.1:8080", "http://unite4ai.com"},
		{"other loopback name", nil, "127.0.0.1:8080", "http://localhost:8080"},
		{"other port on this machine", nil, "127.0.0.1:8080", "http://127.0.0.1:3000"},
		{"https page claiming our host", nil, "127.0.0.1:8080", "https://127.0.0.1:8080"},
		{"opaque origin", nil, "127.0.0.1:8080", "null"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			headers := map[string]string{"Host": tc.host, "Origin": tc.origin}
			want := "origin not allowed: " + tc.origin + "; start the server with --allowed-origins " + tc.origin
			for _, req := range []struct{ method, path, body string }{
				{"POST", "/v1/chat/completions", chatRequest},
				{"GET", "/health", ""},
				{"GET", "/v1/models", ""},
				{"OPTIONS", "/v1/chat/completions", ""},
			} {
				h := newHarness(t, scripted(countPieces...), allowOrigins(tc.allowed...))
				got := h.do(req.method, req.path, req.body, headers)
				if msg := assertRejected(t, h, got, 403); msg != want {
					t.Errorf("%s %s: error = %q, want %q", req.method, req.path, msg, want)
				}
				// The page may read the refusal, so that it can tell the
				// visitor which flag to add, and nothing more.
				wantCORS := map[string]string{"Access-Control-Allow-Origin": tc.origin, "Vary": "Origin"}
				if cors := corsHeaders(got); !reflect.DeepEqual(cors, wantCORS) {
					t.Errorf("%s %s: CORS headers = %v, want %v", req.method, req.path, cors, wantCORS)
				}
			}
		})
	}
}

func TestRecordedPreflightFromUnlistedOriginIsForbidden(t *testing.T) {
	h := newHarness(t)
	got := h.replay(loadExchange(t, "options_chat"))
	assertRejected(t, h, got, 403)
	for _, name := range []string{"Access-Control-Allow-Methods", "Access-Control-Allow-Headers"} {
		if v := got.Header().Get(name); v != "" {
			t.Errorf("%s = %q, want it unset", name, v)
		}
	}
}

func TestPostMustBeJSON(t *testing.T) {
	// A page can send these content types across origins without a preflight,
	// which would let any site drive the model.
	rejected := []string{"", "text/plain", "text/plain; charset=utf-8", "application/x-www-form-urlencoded", "multipart/form-data; boundary=x", "application/jsonx", "application/json, text/plain", "json"}
	for _, contentType := range rejected {
		t.Run("rejected "+contentType, func(t *testing.T) {
			for _, path := range []string{"/v1/chat/completions", "/v1/completions", "/v1/nope"} {
				h := newHarness(t, scripted(countPieces...))
				got := h.do("POST", path, chatRequest, map[string]string{"Content-Type": contentType})
				if msg := assertRejected(t, h, got, 415); !strings.Contains(msg, "application/json") {
					t.Errorf("%s: error = %q, want it to name application/json", path, msg)
				}
			}
		})
	}

	accepted := []string{"application/json", "application/json; charset=utf-8", "application/json;charset=UTF-8", "Application/JSON"}
	for _, contentType := range accepted {
		t.Run("accepted "+contentType, func(t *testing.T) {
			h := newHarness(t, scripted(countPieces...))
			got := h.do("POST", "/v1/chat/completions", chatRequest, map[string]string{"Content-Type": contentType})
			if got.Code != 200 {
				t.Fatalf("status = %d, want 200: %s", got.Code, got.Body)
			}
		})
	}
}

func TestCORSHeadersForAllowedCrossOriginRequests(t *testing.T) {
	requests := []struct {
		name, method, path, body string
		status                   int
	}{
		{"chat", "POST", "/v1/chat/completions", chatRequest, 200},
		{"stream", "POST", "/v1/chat/completions", `{"messages": [], "stream": true}`, 200},
		{"models", "GET", "/v1/models", "", 200},
		{"bad request", "POST", "/v1/chat/completions", `{}`, 400},
		{"unknown path", "GET", "/v1/nope", "", 404},
	}
	configs := []struct {
		name    string
		allowed []string
		want    map[string]string
	}{
		{"listed origin", []string{chatPage}, map[string]string{"Access-Control-Allow-Origin": chatPage, "Vary": "Origin"}},
		{"any origin", []string{"*"}, map[string]string{"Access-Control-Allow-Origin": "*"}},
	}
	for _, cfg := range configs {
		for _, req := range requests {
			t.Run(cfg.name+" "+req.name, func(t *testing.T) {
				h := newHarness(t, scripted(countPieces...), allowOrigins(cfg.allowed...))
				got := h.do(req.method, req.path, req.body, map[string]string{"Origin": chatPage})
				if got.Code != req.status {
					t.Fatalf("status = %d, want %d: %s", got.Code, req.status, got.Body)
				}
				if cors := corsHeaders(got); !reflect.DeepEqual(cors, cfg.want) {
					t.Errorf("CORS headers = %v, want %v", cors, cfg.want)
				}
			})
		}
	}
}

func TestNoCORSHeadersWithoutACrossOriginRequest(t *testing.T) {
	origins := map[string]string{"no origin": "", "same origin": "http://127.0.0.1:8080"}
	for _, allowed := range [][]string{nil, {chatPage}, {"*"}} {
		for name, origin := range origins {
			t.Run(fmt.Sprint(name, " allowing ", allowed), func(t *testing.T) {
				h := newHarness(t, scripted(countPieces...), allowOrigins(allowed...))
				headers := map[string]string{}
				if origin != "" {
					headers["Origin"] = origin
				}
				for _, got := range []*httptest.ResponseRecorder{
					h.do("POST", "/v1/chat/completions", chatRequest, headers),
					h.do("GET", "/health", "", headers),
					h.do("OPTIONS", "/v1/chat/completions", "", headers),
				} {
					if cors := corsHeaders(got); len(cors) != 0 {
						t.Errorf("CORS headers = %v, want none", cors)
					}
				}
			})
		}
	}
}

func TestPreflightFromAllowedOrigin(t *testing.T) {
	const asked = "content-type, authorization, x-stainless-lang"
	cases := []struct {
		name    string
		allowed []string
		headers map[string]string
		want    map[string]string
	}{
		{
			"listed origin", []string{chatPage},
			map[string]string{"Origin": chatPage, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": asked},
			map[string]string{"Access-Control-Allow-Origin": chatPage, "Vary": "Origin", "Access-Control-Allow-Methods": "GET, POST, OPTIONS", "Access-Control-Allow-Headers": asked},
		},
		{
			"any origin", []string{"*"},
			map[string]string{"Origin": chatPage, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": asked},
			map[string]string{"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "GET, POST, OPTIONS", "Access-Control-Allow-Headers": asked},
		},
		{
			"no headers asked for", []string{chatPage},
			loadExchange(t, "options_chat").Request.Headers,
			map[string]string{"Access-Control-Allow-Origin": chatPage, "Vary": "Origin", "Access-Control-Allow-Methods": "GET, POST, OPTIONS"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h := newHarness(t, allowOrigins(tc.allowed...))
			got := h.do("OPTIONS", "/v1/chat/completions", "", tc.headers)
			if got.Code != 204 || got.Body.Len() != 0 {
				t.Fatalf("status = %d with body %q, want 204 and no body", got.Code, got.Body)
			}
			if cors := corsHeaders(got); !reflect.DeepEqual(cors, tc.want) {
				t.Errorf("CORS headers = %v, want %v", cors, tc.want)
			}
			if n := h.engine.Calls.Load(); n != 0 {
				t.Errorf("engine was called %d times", n)
			}
		})
	}
}

func TestOptionsWithoutOriginIsAnEmptyAnswer(t *testing.T) {
	got := newHarness(t).do("OPTIONS", "/v1/chat/completions", "", nil)
	if got.Code != 204 || got.Body.Len() != 0 {
		t.Fatalf("status = %d with body %q, want 204 and no body", got.Code, got.Body)
	}
}
