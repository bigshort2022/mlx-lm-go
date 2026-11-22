package server

import (
	"encoding/json"
	"slices"
	"testing"
)

// fakeTokenizer is a Tokenizer whose detokenizers turn token id i into
// pieces[i], so a test can replay the exact piece sequence of a recording.
// Encode produces one token per byte, which makes prompts easy to assert.
type fakeTokenizer struct {
	pieces []string
	tail   string // returned by every detokenizer's Flush
}

func (f *fakeTokenizer) Encode(text string) []int32 {
	ids := make([]int32, len(text))
	for i := range len(text) {
		ids[i] = int32(text[i])
	}
	return ids
}

func (f *fakeTokenizer) NewDetokenizer() Detokenizer { return fakeDetokenizer{f} }

type fakeDetokenizer struct{ tok *fakeTokenizer }

func (d fakeDetokenizer) Add(token int32) string { return d.tok.pieces[token] }
func (d fakeDetokenizer) Flush() string          { return d.tok.tail }

// fakeTemplater is a Templater that records its arguments and returns a fixed
// prompt or error.
type fakeTemplater struct {
	out string
	err error

	messages, tools, kwargs string
	addGenerationPrompt     bool
}

func (f *fakeTemplater) Render(messages, tools, kwargs json.RawMessage, addGenerationPrompt bool) (string, error) {
	f.messages, f.tools, f.kwargs = string(messages), string(tools), string(kwargs)
	f.addGenerationPrompt = addGenerationPrompt
	return f.out, f.err
}

func TestFakeTokenizerReplaysPieces(t *testing.T) {
	tok := &fakeTokenizer{pieces: []string{"1", ",", " 2"}, tail: "!"}
	var _ Tokenizer = tok
	d := tok.NewDetokenizer()
	got := []string{d.Add(0), d.Add(1), d.Add(2), d.Flush()}
	want := []string{"1", ",", " 2", "!"}
	if !slices.Equal(got, want) {
		t.Fatalf("pieces = %q, want %q", got, want)
	}
}

func TestFakeTokenizerEncodesOneTokenPerByte(t *testing.T) {
	tok := &fakeTokenizer{}
	if got, want := tok.Encode("hi"), []int32{'h', 'i'}; !slices.Equal(got, want) {
		t.Fatalf("Encode = %v, want %v", got, want)
	}
}
