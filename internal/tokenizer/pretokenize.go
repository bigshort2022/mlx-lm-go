package tokenizer

import (
	"fmt"
	"iter"
	"math"
	"time"

	"github.com/dlclark/regexp2"
	"golang.org/x/text/unicode/norm"
)

// normalize applies the tokenizer.json "NFC" normalizer.
//
// The reference normalises with Unicode tables years older than x/text's and
// without x/text's stream-safe limit, so two kinds of text come out
// differently here: combining marks and compositions that Unicode added after
// the reference's tables were made, when they sit next to other marks, and
// runs of more than 30 combining marks, into which x/text inserts U+034F.
func normalize(text string) string {
	return norm.NFC.String(text)
}

// splitter is the tokenizer.json "Split" pre-tokenizer with behaviour
// "Isolated". Its pattern needs a lookahead, which Go's regexp rejects, so it
// runs on regexp2.
//
// The reference runs the pattern on Oniguruma. regexp2 splits the same way
// except where the two engines' data differ: characters newer than
// Oniguruma's Unicode tables fall in other classes, and under (?i) Oniguruma
// also takes U+017F (long s) for an s, which regexp2 does not.
type splitter struct {
	pattern *regexp2.Regexp
}

// newSplitter compiles the split pattern taken from tokenizer.json.
func newSplitter(pattern string) (splitter, error) {
	re, err := regexp2.Compile(pattern, regexp2.None)
	if err != nil {
		return splitter{}, fmt.Errorf("compiling split pattern %q: %w", pattern, err)
	}
	// A timeout is the only way a regexp2 match can fail. Rule it out, whatever
	// regexp2.DefaultMatchTimeout was when the pattern was compiled.
	re.MatchTimeout = time.Duration(math.MaxInt64)
	return splitter{pattern: re}, nil
}

// pieces yields text cut at both ends of every match of the pattern: each
// match is a piece, and so is each stretch of text between matches.
func (s splitter) pieces(text []rune) iter.Seq[[]rune] {
	return func(yield func([]rune) bool) {
		pos := 0
		m, err := s.pattern.FindRunesMatch(text)
		for ; m != nil; m, err = s.pattern.FindNextMatch(m) {
			end := m.Index + m.Length
			if m.Index > pos && !yield(text[pos:m.Index]) {
				return
			}
			if m.Length > 0 && !yield(text[m.Index:end]) {
				return
			}
			pos = end
		}
		if err != nil {
			panic(fmt.Sprintf("tokenizer: split pattern failed with timeouts disabled: %v", err))
		}
		if pos < len(text) {
			yield(text[pos:])
		}
	}
}
