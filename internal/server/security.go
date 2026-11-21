package server

import (
	"fmt"
	"mime"
	"net"
	"net/http"
	"slices"
	"strconv"
	"strings"
)

// The server runs on the user's own machine, where any web page the user
// visits can send it requests. admit and the helpers below are what keeps
// such pages from using the model unless the user allowed their origin.

// admit applies the browser-safety rules to a request and answers CORS
// preflights. When it returns false it has already answered the client.
func (s *server) admit(w http.ResponseWriter, r *http.Request) bool {
	if s.hosts != nil && !s.hosts[strings.ToLower(r.Host)] {
		writeError(w, http.StatusForbidden, "host not allowed: "+r.Host)
		return false
	}
	crossOrigin, ok := s.admitOrigin(w, r)
	if !ok {
		return false
	}
	if r.Method == http.MethodOptions {
		if crossOrigin {
			allowPreflight(w, r)
		}
		w.WriteHeader(http.StatusNoContent)
		return false
	}
	// Only a JSON POST makes a browser ask permission with a preflight before
	// sending a request across origins; form and text posts go straight out.
	if r.Method == http.MethodPost && !isJSON(r.Header.Get("Content-Type")) {
		writeError(w, http.StatusUnsupportedMediaType, "Content-Type must be application/json")
		return false
	}
	return true
}

// admitOrigin refuses requests from pages of origins that are not allowed,
// and sets the CORS headers that let a page of another origin read the
// response. Requests without an Origin, or from the server's own origin, are
// not cross-origin and get no CORS headers.
func (s *server) admitOrigin(w http.ResponseWriter, r *http.Request) (crossOrigin, ok bool) {
	origin := r.Header.Get("Origin")
	if origin == "" || origin == "http://"+strings.ToLower(r.Host) {
		return false, true // not sent by a page, or sent by a page this server serves
	}
	if slices.Contains(s.cfg.AllowedOrigins, "*") {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		return true, true
	}
	// A refusal is readable too, so that the page can tell its visitor which
	// flag is missing. That grants nothing else.
	w.Header().Set("Access-Control-Allow-Origin", origin)
	w.Header().Add("Vary", "Origin")
	if !slices.Contains(s.cfg.AllowedOrigins, origin) {
		writeError(w, http.StatusForbidden, fmt.Sprintf(
			"origin not allowed: %s; start the server with --allowed-origins %s", origin, origin))
		return true, false
	}
	return true, true
}

// allowPreflight tells the browser which methods and headers the page's real
// request may use: the server's methods and whatever headers it asked for.
func allowPreflight(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
	if asked := r.Header.Get("Access-Control-Request-Headers"); asked != "" {
		w.Header().Set("Access-Control-Allow-Headers", asked)
	}
}

// isJSON reports whether a Content-Type header names application/json.
func isJSON(contentType string) bool {
	mediaType, _, err := mime.ParseMediaType(contentType)
	return err == nil && mediaType == "application/json"
}

// allowedHosts returns the Host header values a request may carry when the
// server listens on a loopback address: the names a local client uses for it.
// A page that reaches the server through a DNS name rebound to 127.0.0.1
// carries its own name instead. It returns nil, for no check, when the
// server listens on other addresses, whose names cannot be known.
func allowedHosts(cfg Config) map[string]bool {
	if !isLoopback(cfg.Host) {
		return nil
	}
	hosts := map[string]bool{}
	for _, name := range []string{"127.0.0.1", "localhost", "::1", strings.ToLower(cfg.Host)} {
		hosts[net.JoinHostPort(name, strconv.Itoa(cfg.Port))] = true
		if cfg.Port == 80 { // browsers leave the default port out
			hosts[bracketIPv6(name)] = true
		}
	}
	return hosts
}

func isLoopback(host string) bool {
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// bracketIPv6 writes an IPv6 address the way URLs and Host headers do.
func bracketIPv6(name string) string {
	if strings.Contains(name, ":") {
		return "[" + name + "]"
	}
	return name
}
