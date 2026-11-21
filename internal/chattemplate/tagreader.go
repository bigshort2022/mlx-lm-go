package chattemplate

import "fmt"

// tagReader walks the tokens of one tag.
type tagReader struct {
	tokens []token
	pos    int
	line   int // of the tag, for errors at its end
}

func newTagReader(seg segment) *tagReader {
	return &tagReader{tokens: seg.tokens, line: seg.line}
}

// peek returns the current token, or a tokEnd token past the last one.
func (r *tagReader) peek() token {
	if r.pos < len(r.tokens) {
		return r.tokens[r.pos]
	}
	return token{kind: tokEnd, line: r.line}
}

func (r *tagReader) next() token {
	tok := r.peek()
	if r.pos < len(r.tokens) {
		r.pos++
	}
	return tok
}

// followedBy reports whether the token after the current one is the given
// one.
func (r *tagReader) followedBy(kind tokenKind, text string) bool {
	return r.pos+1 < len(r.tokens) && r.tokens[r.pos+1].kind == kind && r.tokens[r.pos+1].text == text
}

// keyword returns the statement name the tag started with.
func (r *tagReader) keyword() string {
	return r.tokens[0].text
}

func (r *tagReader) isOp(op string) bool {
	tok := r.peek()
	return tok.kind == tokOperator && tok.text == op
}

func (r *tagReader) isName(name string) bool {
	tok := r.peek()
	return tok.kind == tokName && tok.text == name
}

// acceptOp consumes the current token if it is the given operator.
func (r *tagReader) acceptOp(op string) bool {
	if r.isOp(op) {
		r.pos++
		return true
	}
	return false
}

// acceptName consumes the current token if it is the given name.
func (r *tagReader) acceptName(name string) bool {
	if r.isName(name) {
		r.pos++
		return true
	}
	return false
}

func (r *tagReader) expectOp(op string) error {
	if !r.acceptOp(op) {
		return r.errorf("expected %q, found %s", op, r.peek().describe())
	}
	return nil
}

// expectEnd fails unless every token of the tag has been consumed.
func (r *tagReader) expectEnd() error {
	if tok := r.peek(); tok.kind != tokEnd {
		return r.errorf("unexpected %s", tok.describe())
	}
	return nil
}

func (r *tagReader) errorf(format string, args ...any) error {
	return fmt.Errorf("line %d: %s", r.peek().line, fmt.Sprintf(format, args...))
}

// describe names a token for an error message.
func (t token) describe() string {
	switch t.kind {
	case tokEnd:
		return "end of tag"
	case tokString:
		return fmt.Sprintf("string %q", t.text)
	}
	return fmt.Sprintf("%q", t.text)
}
