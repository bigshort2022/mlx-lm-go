package server

import "strings"

// startsInReasoning matches Python's comparison of the last start and end
// token positions by stopping as soon as either is found from the right.
func (t ThinkingTokens) startsInReasoning(prompt []int32) bool {
	for i := len(prompt) - 1; i >= 0; i-- {
		switch prompt[i] {
		case t.End:
			return false
		case t.Start:
			return true
		}
	}
	return false
}

// thinkingSplitter strips the transition marker for the current state and
// emits the surrounding text in order. Only a prefix of the current marker
// may straddle pieces; storing its length avoids retaining any input text.
type thinkingSplitter struct {
	reasoning bool
	pending   int
	literal   *bool // once a tool call begins, its arguments may contain thinking tags
}

func (s *thinkingSplitter) marker() string {
	if s.reasoning {
		return "</think>"
	}
	return "<think>"
}

func (s *thinkingSplitter) add(piece string, emit func(string, bool) error) error {
	if s.literal != nil && *s.literal {
		return emit(piece, false)
	}
	if s.pending != 0 {
		marker := s.marker()
		n := 0
		for n < len(piece) && s.pending < len(marker) && piece[n] == marker[s.pending] {
			n++
			s.pending++
		}
		piece = piece[n:]
		switch {
		case s.pending == len(marker):
			s.reasoning = !s.reasoning
		case piece == "":
			return nil // still a partial marker
		default:
			if err := emit(marker[:s.pending], s.reasoning); err != nil {
				return err
			}
		}
		s.pending = 0
	}

	for piece != "" {
		marker := s.marker()
		before, after, found := strings.Cut(piece, marker)
		if before != "" {
			if !found {
				// Both markers have '<' only at their start.
				if at := strings.LastIndexByte(before, '<'); at >= 0 && strings.HasPrefix(marker, before[at:]) {
					s.pending = len(before) - at
					before = before[:at]
				}
			}
			if before != "" {
				if err := emit(before, s.reasoning); err != nil {
					return err
				}
			}
		}
		if !found {
			break
		}
		if s.literal != nil && *s.literal {
			if err := emit(marker, false); err != nil {
				return err
			}
			return emit(after, false)
		}
		s.reasoning = !s.reasoning
		piece = after
	}
	return nil
}

// flush releases a partial marker at the generation length limit. A stop or
// cancellation instead drops that prefix, as in the reference.
func (s *thinkingSplitter) flush(emit func(string, bool) error) error {
	if s.pending == 0 {
		return nil
	}
	text := s.marker()[:s.pending]
	s.pending = 0
	return emit(text, s.reasoning)
}
