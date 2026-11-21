package server

import (
	"context"
	"fmt"
	"mime"
	"net"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"time"
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
// and sets the CORS headers that let a page of an allowed other origin read
// the response. Requests without an Origin, or from the server's own origin,
// are not cross-origin and get no CORS headers.
func (s *server) admitOrigin(w http.ResponseWriter, r *http.Request) (crossOrigin, ok bool) {
	origin := r.Header.Get("Origin")
	if origin == "" || s.sameOrigin(origin, strings.ToLower(r.Host)) {
		return false, true // not sent by a page, or sent by a page this server serves
	}
	switch {
	case slices.Contains(s.cfg.AllowedOrigins, "*"):
		w.Header().Set("Access-Control-Allow-Origin", "*")
	case slices.Contains(s.cfg.AllowedOrigins, origin):
		w.Header().Set("Access-Control-Allow-Origin", origin)
		w.Header().Add("Vary", "Origin")
	default:
		// No CORS header here: the page must not be able to read the refusal,
		// or any website could find out that this server is running. The
		// message is for clients that are not browsers.
		writeError(w, http.StatusForbidden, fmt.Sprintf(
			"origin not allowed: %s; start the server with --allowed-origins %s", origin, origin))
		return true, false
	}
	return true, true
}

// sameOrigin reports whether origin is that of a page this server itself
// serves to a request with the given Host header.
//
// On loopback, admit has already checked the Host header against the
// server's names. Beyond loopback it cannot, so a page on a DNS name that was
// rebound to this server would pass for the server's own. There the header
// only counts when it holds something that cannot be rebound: an IP address,
// localhost, or the host name the server was started with.
func (s *server) sameOrigin(origin, host string) bool {
	if origin != "http://"+host {
		return false
	}
	if s.hosts != nil {
		return true
	}
	name := hostName(host)
	return net.ParseIP(name) != nil || name == "localhost" ||
		(s.cfg.Host != "" && name == strings.ToLower(s.cfg.Host))
}

// hostName returns the host of a Host header without port and brackets.
func hostName(host string) string {
	if name, _, err := net.SplitHostPort(host); err == nil {
		return name
	}
	return strings.Trim(host, "[]")
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

// resolveTimeout bounds the lookup of the configured host name at startup.
const resolveTimeout = 2 * time.Second

// isLoopback reports whether the server, listening on host, can only be
// reached from this machine: host is a loopback address, or a name whose
// addresses are all loopback, such as localhost, localhost. or localhost6.
func isLoopback(host string) bool {
	if ip := net.ParseIP(host); ip != nil {
		return ip.IsLoopback()
	}
	if strings.EqualFold(host, "localhost") {
		return true // without depending on a resolver
	}
	if host == "" {
		return false // every interface
	}
	ctx, cancel := context.WithTimeout(context.Background(), resolveTimeout)
	defer cancel()
	addrs, err := net.DefaultResolver.LookupIPAddr(ctx, host)
	if err != nil || len(addrs) == 0 {
		return false
	}
	for _, addr := range addrs {
		if !addr.IP.IsLoopback() {
			return false
		}
	}
	return true
}

// bracketIPv6 writes an IPv6 address the way URLs and Host headers do.
func bracketIPv6(name string) string {
	if strings.Contains(name, ":") {
		return "[" + name + "]"
	}
	return name
}
