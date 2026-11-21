package main

import (
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"strconv"
	"sync"
	"time"

	"github.com/bigshort2022/mlx-lm-go/internal/server"
	"github.com/bigshort2022/mlx-lm-go/internal/webui"
)

// runServer serves a model over HTTP until ctx ends.
func runServer(ctx context.Context, args []string, stderr io.Writer) int {
	cfg, err := server.ParseFlags(args)
	if err != nil {
		return usageError(stderr, "server: "+err.Error())
	}
	if err := serve(ctx, cfg, stderr); err != nil {
		return failure(stderr, err)
	}
	return exitOK
}

// shutdownGrace is how long the server waits for its handlers to finish
// after it was told to stop.
const shutdownGrace = 3 * time.Second

// serve loads the model, answers requests until ctx ends, then stops
// accepting connections, cancels the generations in flight and closes the
// engine.
func serve(ctx context.Context, cfg server.Config, stderr io.Writer) error {
	m, err := openModel(cfg.Model, cfg.ChatTemplate)
	if err != nil {
		return err
	}
	// Listen before loading the weights: a port that is taken is then
	// reported at once, and the handler needs the port that was bound.
	listener, err := net.Listen("tcp", net.JoinHostPort(cfg.Host, strconv.Itoa(cfg.Port)))
	if err != nil {
		return err
	}
	defer listener.Close()
	addr := listener.Addr().(*net.TCPAddr)
	cfg.Port = addr.Port

	eng, err := m.loadEngine(cfg.PrefillStepSize)
	if err != nil {
		return err
	}
	defer eng.Close()
	if ctx.Err() != nil {
		return nil // told to stop while the model was loading
	}

	handler := server.New(cfg, server.Deps{
		Engine:    countingEngine{eng},
		Tokenizer: serverTokenizer{m.tokenizer},
		Template:  serverTemplate{m.template},
		WebUI:     webui.FS(),
	})
	var unused unusedConns
	srv := &http.Server{
		Handler: logRequests(newLogger(stderr, cfg.LogLevel), handler),
		// Every request ends with ctx, and with the request its generation.
		BaseContext: func(net.Listener) context.Context { return ctx },
		ConnState:   unused.track,
	}
	srv.RegisterOnShutdown(unused.close)
	served := make(chan error, 1)
	go func() { served <- srv.Serve(listener) }()
	fmt.Fprintf(stderr, "listening on http://%s\n", net.JoinHostPort(cfg.Host, strconv.Itoa(cfg.Port)))
	if warning := exposureWarning(addr); warning != "" {
		fmt.Fprintln(stderr, warning)
	}

	select {
	case err := <-served:
		return err
	case <-ctx.Done():
	}
	// The handlers return once their generations are cancelled. A client
	// that then does not take the end of its response is cut off.
	grace, cancel := context.WithTimeout(context.Background(), shutdownGrace)
	defer cancel()
	if err := srv.Shutdown(grace); err != nil {
		srv.Close()
	}
	return nil
}

// exposureWarning returns what to tell the user when the server listens on
// addr, or "" if addr can only be reached from this machine.
func exposureWarning(addr *net.TCPAddr) string {
	if addr.IP.IsLoopback() {
		return ""
	}
	return fmt.Sprintf("warning: listening on %s, which other machines can reach: "+
		"whoever can connect can use the model", addr)
}

// unusedConns are the connections that have not sent a request yet. Browsers
// open such connections ahead of need, and http.Server.Shutdown waits up to
// five seconds for each to send one, so the server closes them itself when
// it shuts down.
type unusedConns struct {
	mu    sync.Mutex
	conns map[net.Conn]struct{}
}

// track is the http.Server.ConnState hook.
func (u *unusedConns) track(conn net.Conn, state http.ConnState) {
	u.mu.Lock()
	defer u.mu.Unlock()
	if state != http.StateNew {
		delete(u.conns, conn)
		return
	}
	if u.conns == nil {
		u.conns = map[net.Conn]struct{}{}
	}
	u.conns[conn] = struct{}{}
}

// close closes the connections that are still unused.
func (u *unusedConns) close() {
	u.mu.Lock()
	defer u.mu.Unlock()
	for conn := range u.conns {
		conn.Close()
	}
}
