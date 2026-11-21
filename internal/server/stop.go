package server

import "strings"

// stopMatcher watches generated text for stop strings. Text arrives in the
// pieces the detokenizer produces. A piece is held back while its end could be
// the start of a stop string, so that no part of a stop string is ever sent.
//
// Like the reference, which blanks the text of every token of a matched stop
// sequence, a match drops each piece it touches as a whole. This is why
// "1, 2, 3" stopped at "3" ends as "1, 2," when the 3 arrived as " 3".
type stopMatcher struct {
	stops []string
	held  []string // pieces not yet released, oldest first
}

// newStopMatcher returns a matcher for the given stop strings. Empty strings
// are ignored, since they would match everywhere.
func newStopMatcher(stops []string) *stopMatcher {
	m := &stopMatcher{}
	for _, stop := range stops {
		if stop != "" {
			m.stops = append(m.stops, stop)
		}
	}
	return m
}

// add takes the next piece of generated text. It returns the text that may be
// sent to the client now and whether a stop string has been completed. After a
// stop the matcher holds nothing.
func (m *stopMatcher) add(piece string) (release string, stopped bool) {
	if len(m.stops) == 0 || piece == "" {
		return piece, false
	}
	m.held = append(m.held, piece)
	text := strings.Join(m.held, "")
	if at, found := m.find(text); found {
		release = m.take(at)
		m.held = nil
		return release, true
	}
	return m.take(len(text) - m.partial(text)), false
}

// flush returns the text still held back when generation ends.
func (m *stopMatcher) flush() string {
	rest := strings.Join(m.held, "")
	m.held = nil
	return rest
}

// find returns the offset of the earliest stop string in text.
func (m *stopMatcher) find(text string) (at int, found bool) {
	for _, stop := range m.stops {
		if i := strings.Index(text, stop); i >= 0 && (!found || i < at) {
			at, found = i, true
		}
	}
	return at, found
}

// partial returns the length of the longest end of text that is the beginning
// of a stop string, and so might become one with the next pieces.
func (m *stopMatcher) partial(text string) int {
	longest := 0
	for _, stop := range m.stops {
		for n := min(len(stop)-1, len(text)); n > longest; n-- {
			if strings.HasSuffix(text, stop[:n]) {
				longest = n
				break
			}
		}
	}
	return longest
}

// take removes the held pieces that end at or before offset end and returns
// their text.
func (m *stopMatcher) take(end int) string {
	n, size := 0, 0
	for n < len(m.held) && size+len(m.held[n]) <= end {
		size += len(m.held[n])
		n++
	}
	taken := strings.Join(m.held[:n], "")
	m.held = m.held[n:]
	return taken
}
