package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"io/fs"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/bigshort2022/mlx-lm-go/internal/server"
	"github.com/bigshort2022/mlx-lm-go/internal/webui"
)

// startTimeout bounds how long a test waits for the server to come up or to
// go down.
const startTimeout = 30 * time.Second

// output collects what a running command writes.
type output struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (o *output) Write(p []byte) (int, error) {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.buf.Write(p)
}

func (o *output) String() string {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.buf.String()
}

// liveServer is a server command running in the test process.
type liveServer struct {
	url    string  // base URL, such as http://127.0.0.1:49152
	stderr *output // what the command has written to stderr so far
	cancel context.CancelFunc
	exit   chan int // holds the exit code once the command has returned
}

var listeningLine = regexp.MustCompile(`(?m)^listening on (http://\S+)$`)

// startServer runs "server --model <reference model> --port 0" with the given
// extra flags and waits until it listens. The server is shut down when the
// test ends.
func startServer(t *testing.T, flags ...string) *liveServer {
	t.Helper()
	needModel(t)
	args := append([]string{"server", "--model", referenceModel, "--port", "0"}, flags...)
	return startServing(t, func(ctx context.Context, stderr io.Writer) int {
		return run(ctx, args, io.Discard, stderr)
	})
}

// startServing runs command until its context ends, waits until it listens,
// and shuts it down when the test ends.
func startServing(t *testing.T, command func(ctx context.Context, stderr io.Writer) int) *liveServer {
	t.Helper()

	ctx, cancel := context.WithCancel(context.Background())
	exit := make(chan int, 1)
	s := &liveServer{stderr: &output{}, cancel: cancel, exit: exit}
	go func() { exit <- command(ctx, s.stderr) }()
	t.Cleanup(func() {
		cancel()
		select {
		case <-exit:
		case <-time.After(startTimeout):
			t.Error("the server did not shut down in time")
		}
	})

	deadline := time.After(startTimeout)
	for {
		if m := listeningLine.FindStringSubmatch(s.stderr.String()); m != nil {
			s.url = m[1]
			return s
		}
		select {
		case code := <-exit:
			exit <- code // for the cleanup
			t.Fatalf("the server exited with code %d before it listened; stderr:\n%s", code, s.stderr)
		case <-deadline:
			t.Fatalf("the server did not listen in time; stderr:\n%s", s.stderr)
		case <-time.After(5 * time.Millisecond):
		}
	}
}

// exchange is one request/response pair recorded from Python mlx_lm.server,
// read from testdata/server at the repository root.
type exchange struct {
	Request struct {
		Path string          `json:"path"`
		Body json.RawMessage `json:"body"`
	} `json:"request"`
	Response struct {
		Body string `json:"body"`
	} `json:"response"`
}

func loadExchange(t *testing.T, name string) exchange {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "server", name+".json"))
	if err != nil {
		t.Fatal(err)
	}
	var e exchange
	if err := json.Unmarshal(raw, &e); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	return e
}

// completion is what the tests compare of a chat or text completion.
type completion struct {
	Choices []struct {
		FinishReason string `json:"finish_reason"`
		Message      struct {
			Content string `json:"content"`
		} `json:"message"`
		Text string `json:"text"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
}

func decodeCompletion(t *testing.T, body string) completion {
	t.Helper()
	var c completion
	if err := json.Unmarshal([]byte(body), &c); err != nil {
		t.Fatalf("%v in body:\n%s", err, body)
	}
	if len(c.Choices) != 1 {
		t.Fatalf("got %d choices, want 1:\n%s", len(c.Choices), body)
	}
	return c
}

// post sends a JSON request and returns the response body, which must come
// with status 200.
func (s *liveServer) post(t *testing.T, path, body string) string {
	t.Helper()
	resp, err := http.Post(s.url+path, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	got, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("POST %s: status %d, want 200; body:\n%s", path, resp.StatusCode, got)
	}
	return string(got)
}

// replay sends the request of a recording to the server and returns the
// recorded completion and the server's.
func (s *liveServer) replay(t *testing.T, name string) (want, got completion) {
	t.Helper()
	rec := loadExchange(t, name)
	want = decodeCompletion(t, rec.Response.Body)
	got = decodeCompletion(t, s.post(t, rec.Request.Path, string(rec.Request.Body)))
	return want, got
}

func TestServerAnswersChatLikeTheRecording(t *testing.T) {
	s := startServer(t)

	want, got := s.replay(t, "chat")
	if got, want := got.Choices[0].Message.Content, want.Choices[0].Message.Content; got != want {
		t.Errorf("content = %q, want %q", got, want)
	}
	if got, want := got.Choices[0].FinishReason, want.Choices[0].FinishReason; got != want {
		t.Errorf("finish_reason = %q, want %q", got, want)
	}
	if got.Usage != want.Usage {
		t.Errorf("usage = %+v, want %+v", got.Usage, want.Usage)
	}
}

// streamed is what the tests compare of a completion stream.
type streamed struct {
	Pieces       []string // the text of each event that carries some
	FinishReason string
}

// decodeStream reads the events of a completion stream. It fails unless the
// stream ends with [DONE].
func decodeStream(t *testing.T, body string) streamed {
	t.Helper()
	var events []string
	for line := range strings.Lines(body) {
		if data, ok := strings.CutPrefix(line, "data: "); ok {
			events = append(events, strings.TrimSuffix(data, "\n"))
		}
	}
	if len(events) == 0 || events[len(events)-1] != "[DONE]" {
		t.Fatalf("the stream does not end with [DONE]:\n%s", body)
	}
	var s streamed
	for _, event := range events[:len(events)-1] {
		var chunk struct {
			Choices []struct {
				FinishReason string `json:"finish_reason"`
				Delta        struct {
					Content string `json:"content"`
				} `json:"delta"`
				Text string `json:"text"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(event), &chunk); err != nil {
			t.Fatalf("%v in event %s", err, event)
		}
		for _, choice := range chunk.Choices {
			if piece := choice.Delta.Content + choice.Text; piece != "" {
				s.Pieces = append(s.Pieces, piece)
			}
			s.FinishReason = choice.FinishReason
		}
	}
	return s
}

// The pieces of a stream are those of the reference's detokenizer, which
// joins a space to the word after it.
func TestServerStreamsLikeTheRecordings(t *testing.T) {
	s := startServer(t)

	for _, name := range []string{"chat_stream", "chat_stream_stop", "chat_stream_usage", "completions_stream"} {
		t.Run(name, func(t *testing.T) {
			rec := loadExchange(t, name)
			want := decodeStream(t, rec.Response.Body)
			got := decodeStream(t, s.post(t, rec.Request.Path, string(rec.Request.Body)))
			if !reflect.DeepEqual(got, want) {
				t.Errorf("stream = %+q, want %+q", got, want)
			}
		})
	}
}

func TestServerAnswersLikeTheRecordings(t *testing.T) {
	s := startServer(t)

	for _, name := range []string{"chat_length", "chat_stop_string", "chat_stop_single", "chat_content_parts", "completions"} {
		t.Run(name, func(t *testing.T) {
			want, got := s.replay(t, name)
			if got, want := got.Choices[0].Message.Content, want.Choices[0].Message.Content; got != want {
				t.Errorf("content = %q, want %q", got, want)
			}
			if got, want := got.Choices[0].Text, want.Choices[0].Text; got != want {
				t.Errorf("text = %q, want %q", got, want)
			}
			if got, want := got.Choices[0].FinishReason, want.Choices[0].FinishReason; got != want {
				t.Errorf("finish_reason = %q, want %q", got, want)
			}
		})
	}
}

// get returns the status and body of a GET request.
func (s *liveServer) get(t *testing.T, path string) (status int, body string) {
	t.Helper()
	resp, err := http.Get(s.url + path)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	return resp.StatusCode, string(raw)
}

func TestServerServesTheEmbeddedUI(t *testing.T) {
	s := startServer(t)

	want, err := fs.ReadFile(webui.FS(), "index.html")
	if err != nil {
		t.Fatal(err)
	}
	status, body := s.get(t, "/")
	if status != http.StatusOK || body != string(want) {
		t.Errorf("GET / = status %d with a body of %d bytes, want 200 with the %d bytes of the UI's index.html",
			status, len(body), len(want))
	}
}

func TestServerListsTheModelItWasStartedWith(t *testing.T) {
	s := startServer(t)

	status, body := s.get(t, "/v1/models")
	var list struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.Unmarshal([]byte(body), &list); err != nil {
		t.Fatalf("%v in body:\n%s", err, body)
	}
	if status != http.StatusOK || len(list.Data) != 1 || list.Data[0].ID != referenceModel {
		t.Errorf("GET /v1/models = status %d listing %+v, want 200 listing only %s", status, list.Data, referenceModel)
	}
}

// stop ends the server's context and returns the exit code of the command.
// It fails the test unless the command returns within five seconds.
func (s *liveServer) stop(t *testing.T) int {
	t.Helper()
	s.cancel()
	select {
	case code := <-s.exit:
		s.exit <- code // for the cleanup
		return code
	case <-time.After(5 * time.Second):
		t.Fatal("the server was still running five seconds after its context ended")
		return 0
	}
}

func TestServerShutsDownDuringAGeneration(t *testing.T) {
	s := startServer(t)
	closesBefore := shared.closeCount()

	resp, err := http.Post(s.url+"/v1/chat/completions", "application/json", strings.NewReader(
		`{"messages": [{"role": "user", "content": "Write a long story."}], "max_tokens": 2000, "stream": true}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	// Generation is under way once the first piece of text has arrived.
	events := bufio.NewReader(resp.Body)
	for {
		line, err := events.ReadString('\n')
		if err != nil {
			t.Fatalf("the stream ended before any text arrived: %v", err)
		}
		if strings.HasPrefix(line, "data: ") {
			break
		}
	}

	if code := s.stop(t); code != 0 {
		t.Errorf("exit code = %d, want 0; stderr:\n%s", code, s.stderr)
	}
	// The server ends the response properly instead of dropping the
	// connection, so the client can tell the end from a network failure.
	rest, err := io.ReadAll(events)
	if err != nil {
		t.Errorf("reading the rest of the stream: %v", err)
	}
	if strings.Contains(string(rest), "[DONE]") {
		t.Error("the generation ran to its end instead of being cancelled")
	}
	if generated := shared.lastResult().CompletionTokens; generated >= 2000 {
		t.Errorf("the engine generated all %d tokens", generated)
	}
	if closed := shared.closeCount() - closesBefore; closed != 1 {
		t.Errorf("the server closed its engine %d times, want once", closed)
	}
}

// Browsers open connections ahead of need. One that has not sent a request
// must not hold the shutdown up.
func TestServerShutsDownAtOnceDespiteAnUnusedConnection(t *testing.T) {
	s := startServer(t)
	conn, err := net.Dial("tcp", strings.TrimPrefix(s.url, "http://"))
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	s.get(t, "/health") // the server is serving, so it has accepted conn too

	start := time.Now()
	if code := s.stop(t); code != 0 {
		t.Errorf("exit code = %d, want 0; stderr:\n%s", code, s.stderr)
	}
	if took := time.Since(start); took > time.Second {
		t.Errorf("the shutdown took %v, want it to be immediate", took.Round(time.Millisecond))
	}
}

// The user may interrupt the command while it still loads the model.
func TestServerStoppedBeforeItIsReadyNeverListens(t *testing.T) {
	needModel(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	var stderr bytes.Buffer
	code := run(ctx, []string{"server", "--model", referenceModel, "--port", "0"}, io.Discard, &stderr)
	if code != 0 || stderr.Len() != 0 {
		t.Errorf("exit code = %d, stderr = %q, want 0 and no output", code, &stderr)
	}
}

func TestSecondServerOnTheSamePortFails(t *testing.T) {
	first := startServer(t)
	address := strings.TrimPrefix(first.url, "http://")
	_, port, err := net.SplitHostPort(address)
	if err != nil {
		t.Fatal(err)
	}
	closesBefore := shared.closeCount()

	code, _, stderr := runCommand(t, "server", "--model", referenceModel, "--port", port)
	if code != 1 {
		t.Errorf("exit code = %d, want 1", code)
	}
	if !strings.Contains(stderr, address) || !strings.Contains(stderr, "address already in use") {
		t.Errorf("stderr does not say that %s is taken:\n%s", address, stderr)
	}
	// The port is tried before the weights are loaded.
	if shared.closeCount() != closesBefore {
		t.Error("the second server loaded the model although it could not listen")
	}
}

func TestServerPassesThePrefillStepSizeToTheEngine(t *testing.T) {
	startServer(t, "--prefill-step-size", "16")

	if got := shared.lastOptions().PrefillStepSize; got != 16 {
		t.Errorf("the engine was loaded with prefill step size %d, want 16", got)
	}
}

// awaitLog returns the first match of pattern in the server's stderr. A
// request is logged after its response has gone out, so the line may come
// later than the response.
func (s *liveServer) awaitLog(t *testing.T, pattern string) string {
	t.Helper()
	re := regexp.MustCompile(pattern)
	deadline := time.Now().Add(3 * time.Second)
	for {
		if line := re.FindString(s.stderr.String()); line != "" {
			return line
		}
		if time.Now().After(deadline) {
			t.Fatalf("stderr has no line matching %s:\n%s", pattern, s.stderr)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func TestServerLogsEachFinishedRequest(t *testing.T) {
	s := startServer(t)

	s.replay(t, "chat")
	s.awaitLog(t, `(?m)^time=\S+ level=INFO msg=request method=POST path=/v1/chat/completions status=200 `+
		`prompt_tokens=37 completion_tokens=14 duration=\S+$`)

	s.get(t, "/v1/models")
	s.awaitLog(t, `(?m)^time=\S+ level=INFO msg=request method=GET path=/v1/models status=200 `+
		`prompt_tokens=0 completion_tokens=0 duration=\S+$`)
}

func TestServerLogsTheStatusOfARefusedRequest(t *testing.T) {
	s := startServer(t)

	resp, err := http.Post(s.url+"/v1/chat/completions", "application/json", strings.NewReader("{"))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	s.awaitLog(t, `(?m) msg=request method=POST path=/v1/chat/completions status=400 prompt_tokens=0 completion_tokens=0 `)
}

func TestServerLogLevelSilencesRequestLines(t *testing.T) {
	s := startServer(t, "--log-level", "WARNING")

	s.replay(t, "chat")
	s.stop(t) // by now every request has been logged
	if stderr := s.stderr.String(); strings.Contains(stderr, "msg=request") {
		t.Errorf("stderr has request lines although the log level is WARNING:\n%s", stderr)
	}
}

func TestServerLogsNeitherPromptsNorCompletions(t *testing.T) {
	s := startServer(t, "--log-level", "DEBUG")

	s.replay(t, "chat")        // "Count from 1 to 5." answered with "1, 2, 3, 4, 5"
	s.replay(t, "completions") // "1, 2, 3," continued with "4, 5, 6"
	rec := loadExchange(t, "chat_stream")
	s.post(t, rec.Request.Path, string(rec.Request.Body))
	s.stop(t) // by now every request has been logged

	stderr := s.stderr.String()
	if got := strings.Count(stderr, "msg=request"); got != 3 {
		t.Errorf("stderr has %d request lines, want 3:\n%s", got, stderr)
	}
	for _, text := range []string{"Count", "1, 2", "4, 5"} {
		if strings.Contains(stderr, text) {
			t.Errorf("stderr contains %q, which is prompt or completion text:\n%s", text, stderr)
		}
	}
}

func TestExposureWarningForAddressesBeyondLoopback(t *testing.T) {
	tests := []struct {
		ip      string
		exposed bool
	}{
		{"127.0.0.1", false},
		{"::1", false},
		{"0.0.0.0", true},
		{"::", true},
		{"192.168.1.20", true},
	}
	for _, tt := range tests {
		t.Run(tt.ip, func(t *testing.T) {
			addr := &net.TCPAddr{IP: net.ParseIP(tt.ip), Port: 8080}
			warning := exposureWarning(addr)
			if !tt.exposed && warning != "" {
				t.Errorf("warning = %q, want none", warning)
			}
			if tt.exposed && !(strings.HasPrefix(warning, "warning: ") && strings.Contains(warning, addr.String())) {
				t.Errorf("warning = %q, want one that names %s", warning, addr)
			}
		})
	}
}

func TestServerOnLoopbackStartsWithoutAWarning(t *testing.T) {
	s := startServer(t)

	if got, want := s.stderr.String(), "listening on "+s.url+"\n"; got != want {
		t.Errorf("stderr = %q, want %q", got, want)
	}
}

func TestServerChatTemplateFlagReplacesTheModelsTemplate(t *testing.T) {
	const text = "Count from 1 to 5."
	const request = `{"messages": [{"role": "user", "content": "` + text + `"}], "max_tokens": 1}`
	tok := referenceTokenizer(t)
	const imEnd = 151645 // <|im_end|>, the eos_token of tokenizer_config.json

	tests := []struct {
		name     string
		template string
		prompt   []int32
	}{
		{"the user text alone", "{{ messages[0].content }}", tok.Encode(text)},
		{"with a special token of the model", "{{ messages[0].content }}{{ eos_token }}", append(tok.Encode(text), imEnd)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := startServer(t, "--chat-template", tt.template)

			got := decodeCompletion(t, s.post(t, "/v1/chat/completions", request))
			if got.Usage.PromptTokens != len(tt.prompt) {
				t.Errorf("usage.prompt_tokens = %d, want %d", got.Usage.PromptTokens, len(tt.prompt))
			}
			if got := shared.lastRequest().Prompt; !slices.Equal(got, tt.prompt) {
				t.Errorf("the engine received the prompt %v, want %v", got, tt.prompt)
			}
		})
	}
}

func TestServerReportsAChatTemplateItCannotParse(t *testing.T) {
	needModel(t)

	// A server that starts all the same is stopped, and then returns 0.
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	var stderr bytes.Buffer
	code := run(ctx, []string{"server", "--model", referenceModel, "--port", "0", "--chat-template", "{% if %}"}, io.Discard, &stderr)
	if code != 1 {
		t.Errorf("exit code = %d, want 1", code)
	}
	if !strings.HasPrefix(stderr.String(), "mlx-lm-go: --chat-template: ") {
		t.Errorf("stderr does not blame the --chat-template flag:\n%s", &stderr)
	}
}

// A client that stalls in the middle of a request header would otherwise
// hold its connection for ever. The server is given a short timeout here in
// place of readHeaderTimeout.
func TestServerClosesAConnectionThatStallsInARequestHeader(t *testing.T) {
	needModel(t)
	cfg, err := server.ParseFlags([]string{"--model", referenceModel, "--port", "0"})
	if err != nil {
		t.Fatal(err)
	}
	s := startServing(t, func(ctx context.Context, stderr io.Writer) int {
		if err := serve(ctx, cfg, stderr, 100*time.Millisecond); err != nil {
			return failure(stderr, err)
		}
		return exitOK
	})

	conn, err := net.Dial("tcp", strings.TrimPrefix(s.url, "http://"))
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	if _, err := io.WriteString(conn, "GET /hea"); err != nil {
		t.Fatal(err)
	}
	// ReadAll returns without an error once the server has closed the
	// connection, and with one when the deadline passes first.
	if err := conn.SetReadDeadline(time.Now().Add(2 * time.Second)); err != nil {
		t.Fatal(err)
	}
	if _, err := io.ReadAll(conn); err != nil {
		t.Errorf("the server kept the stalled connection open: %v", err)
	}
}
