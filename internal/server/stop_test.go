package server

import (
	"net/http/httptest"
	"reflect"
	"slices"
	"testing"
)

// step is one piece fed to a stopMatcher and what it must answer.
type step struct {
	piece   string
	release string
	stopped bool
}

func checkSteps(t *testing.T, m *stopMatcher, steps ...step) {
	t.Helper()
	for i, s := range steps {
		release, stopped := m.add(s.piece)
		if release != s.release || stopped != s.stopped {
			t.Fatalf("step %d: add(%q) = (%q, %v), want (%q, %v)", i, s.piece, release, stopped, s.release, s.stopped)
		}
	}
}

func TestStopMatcherWithoutStopsPassesTextThrough(t *testing.T) {
	m := newStopMatcher(nil)
	checkSteps(t, m, step{"1", "1", false}, step{",", ",", false}, step{" 2", " 2", false})
	if rest := m.flush(); rest != "" {
		t.Fatalf("flush = %q, want nothing", rest)
	}
}

func TestStopMatcherDropsThePieceThatHoldsTheStopString(t *testing.T) {
	// The recorded exchange for stop ["3"]: the reference blanks the text of
	// the matching tokens, so the space that came with the 3 goes too.
	m := newStopMatcher([]string{"3"})
	checkSteps(t, m,
		step{"1", "1", false},
		step{",", ",", false},
		step{" 2", " 2", false},
		step{",", ",", false},
		step{" 3", "", true},
	)
}

func TestStopMatcherNeverReleasesAStopStringSplitAcrossPieces(t *testing.T) {
	m := newStopMatcher([]string{"34"})
	checkSteps(t, m, step{"12", "12", false}, step{"3", "", false}, step{"4", "", true})
}

func TestStopMatcherReleasesTextOnceTheMatchFails(t *testing.T) {
	m := newStopMatcher([]string{"34"})
	checkSteps(t, m, step{"3", "", false}, step{"5", "35", false})
	if rest := m.flush(); rest != "" {
		t.Fatalf("flush = %q, want nothing", rest)
	}
}

func TestStopMatcherFlushReturnsHeldText(t *testing.T) {
	m := newStopMatcher([]string{"34"})
	checkSteps(t, m, step{"12", "12", false}, step{"3", "", false})
	if rest := m.flush(); rest != "3" {
		t.Fatalf("flush = %q, want %q", rest, "3")
	}
	if rest := m.flush(); rest != "" {
		t.Fatalf("second flush = %q, want nothing", rest)
	}
}

func TestStopMatcherMultiPieceStop(t *testing.T) {
	// The recorded exchange for stop ", 4".
	m := newStopMatcher([]string{", 4"})
	checkSteps(t, m,
		step{"1", "1", false},
		step{",", "", false},
		step{" 2", ", 2", false},
		step{",", "", false},
		step{" 3", ", 3", false},
		step{",", "", false},
		step{" 4", "", true},
	)
}

func TestStopMatcherReleasesHeldPiecesBeforeTheMatch(t *testing.T) {
	m := newStopMatcher([]string{"bcd"})
	checkSteps(t, m, step{"ab", "", false}, step{"b", "ab", false}, step{"cd", "", true})

	m = newStopMatcher([]string{"bcd"})
	checkSteps(t, m, step{"b", "", false}, step{"c", "", false}, step{"bcd", "bc", true})
}

func TestStopMatcherStopsAtTheEarliestOfSeveralStops(t *testing.T) {
	m := newStopMatcher([]string{"world", "lo"})
	checkSteps(t, m, step{"hel", "", false}, step{"lo world", "hel", true})

	m = newStopMatcher([]string{"xyz", "y"})
	checkSteps(t, m, step{"a", "a", false}, step{"x", "", false}, step{"y", "x", true})
}

func TestStopMatcherIgnoresEmptyStopStrings(t *testing.T) {
	m := newStopMatcher([]string{""})
	checkSteps(t, m, step{"a", "a", false}, step{"b", "b", false})
}

func TestChatStopStringsMatchRecordings(t *testing.T) {
	cases := []struct {
		name    string
		content string
		tokens  int // tokens the engine emitted before it was cancelled
	}{
		{"chat_stop_string", "1, 2,", 5},
		{"chat_stop_single", "1, 2, 3", 7},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rec := loadExchange(t, tc.name)
			errs := make(chan error, 1)
			h := newHarness(t, scripted(countPieces...), func(h *harness) { h.wrap = reportErrors(errs) })
			got := h.replay(rec)

			if got.Code != 200 {
				t.Fatalf("status = %d, want 200: %s", got.Code, got.Body)
			}
			if g, w := shapeOf(t, got.Body.String()), shapeOf(t, rec.Response.Body); !reflect.DeepEqual(g, w) {
				t.Fatalf("shape =\n%v, want\n%v", g, w)
			}
			body, want := decodeCompletion(t, got.Body.String()), decodeCompletion(t, rec.Response.Body)
			if want.Choices[0].Message.Content != tc.content {
				t.Fatalf("recording holds %q, test expects %q", want.Choices[0].Message.Content, tc.content)
			}
			if content := body.Choices[0].Message.Content; content != tc.content {
				t.Errorf("content = %q, want %q", content, tc.content)
			}
			if fr := body.finish(t); fr != "stop" {
				t.Errorf("finish_reason = %q, want stop", fr)
			}
			awaitCanceled(t, errs) // the engine did not run to the end of its script
			if body.Usage.CompletionTokens != tc.tokens {
				t.Errorf("completion_tokens = %d, want the %d the engine reported", body.Usage.CompletionTokens, tc.tokens)
			}
		})
	}
}

func TestChatStreamStopMatchesRecording(t *testing.T) {
	rec := loadExchange(t, "chat_stream_stop")
	errs := make(chan error, 1)
	h := newHarness(t, scripted(countPieces...), func(h *harness) { h.wrap = reportErrors(errs) })
	got := h.replay(rec)
	chunks := assertStreamShape(t, got, rec)

	var recorded []completionBody
	for _, data := range sseData(t, rec.Response.Body) {
		if data != "[DONE]" {
			recorded = append(recorded, decodeCompletion(t, data))
		}
	}
	if g, w := deltas(chunks), deltas(recorded); !slices.Equal(g, w) {
		t.Errorf("delta contents = %q, want the recorded %q", g, w)
	}
	if fr := chunks[len(chunks)-1].finish(t); fr != "stop" {
		t.Errorf("final finish_reason = %q, want stop", fr)
	}
	awaitCanceled(t, errs)
}

// streamedText replays a stream and returns its text deltas joined together
// and the final finish reason.
func streamedText(t *testing.T, got *httptest.ResponseRecorder) (text, finish string) {
	t.Helper()
	for _, data := range sseData(t, got.Body.String()) {
		if data == "[DONE]" {
			break
		}
		chunk := decodeCompletion(t, data)
		text += chunk.Choices[0].Delta.Content
		finish = chunk.finish(t)
	}
	return text, finish
}

func TestChatStopStringSplitAcrossPieces(t *testing.T) {
	cases := []struct {
		name   string
		pieces []string
		tail   string // text the detokenizer releases only at the end
		text   string
	}{
		{"completed stop string is never sent", []string{"12", "3", "4", "5"}, "", "12"},
		{"near miss is released", []string{"3", "5"}, "", "35"},
		{"held text is released at the end", []string{"12", "3"}, "", "123"},
		{"stop string completed by the last text", []string{"12", "3"}, "4", "12"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h := newHarness(t, scripted(tc.pieces...), func(h *harness) { h.tok.tail = tc.tail })

			got := h.do("POST", "/v1/chat/completions", `{"messages": [], "stop": "34"}`, nil)
			body := decodeCompletion(t, got.Body.String())
			if content := body.Choices[0].Message.Content; content != tc.text || body.finish(t) != "stop" {
				t.Errorf("not streamed: content = %q, finish_reason = %q, want %q, stop", content, body.finish(t), tc.text)
			}

			got = h.do("POST", "/v1/chat/completions", `{"messages": [], "stop": ["34"], "stream": true}`, nil)
			if text, finish := streamedText(t, got); text != tc.text || finish != "stop" {
				t.Errorf("streamed: text = %q, finish_reason = %q, want %q, stop", text, finish, tc.text)
			}
		})
	}
}
