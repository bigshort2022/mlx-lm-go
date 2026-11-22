package server

import (
	"bytes"
	"io/fs"
	"mime"
	"net/http"
	"path"
	"strings"
	"time"
)

// uiPolicy is the Content-Security-Policy of every UI response. The page may
// load scripts and styles from, and talk to, this server only; it cannot be
// re-based, made to post a form elsewhere, or shown inside another page.
const uiPolicy = "default-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"

// uiContentTypes are the types of the files the UI is made of. They are
// fixed here because the system's MIME table, which mime.TypeByExtension
// consults, differs between machines.
var uiContentTypes = map[string]string{
	".html": "text/html; charset=utf-8",
	".js":   "text/javascript; charset=utf-8",
	".css":  "text/css; charset=utf-8",
}

// serveUI answers a request for a file of the web UI and reports whether it
// did. The UI is the same chat page that unite4ai.com hosts; served from here
// it shares the API's origin, so it works in browsers that keep public pages
// from calling loopback addresses.
func (s *server) serveUI(w http.ResponseWriter, r *http.Request) bool {
	if s.deps.WebUI == nil || (r.Method != http.MethodGet && r.Method != http.MethodHead) {
		return false
	}
	name := strings.TrimPrefix(r.URL.Path, "/")
	if name == "" {
		name = "index.html"
	}
	if !fs.ValidPath(name) {
		return false
	}
	body, err := fs.ReadFile(s.deps.WebUI, name)
	if err != nil {
		return false // no such file, or a directory
	}
	w.Header().Set("Content-Type", uiContentType(name))
	w.Header().Set("Content-Security-Policy", uiPolicy)
	w.Header().Set("X-Content-Type-Options", "nosniff")
	// The files keep their names from build to build, so the browser must
	// ask again rather than run the UI of an older binary.
	w.Header().Set("Cache-Control", "no-cache")
	// ServeContent leaves the body out of HEAD responses. The zero time means
	// the files carry no modification date.
	http.ServeContent(w, r, name, time.Time{}, bytes.NewReader(body))
	return true
}

// uiContentType returns the Content-Type for a UI file.
func uiContentType(name string) string {
	ext := path.Ext(name)
	if contentType, ok := uiContentTypes[ext]; ok {
		return contentType
	}
	if contentType := mime.TypeByExtension(ext); contentType != "" {
		return contentType
	}
	return "application/octet-stream"
}
