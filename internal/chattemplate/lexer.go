package chattemplate

import (
	"fmt"
	"strings"
	"unicode/utf8"
)

// The lexer follows jinja2/lexer.py as configured by Hugging Face:
// trim_blocks and lstrip_blocks on, keep_trailing_newline off.

type segmentKind int

const (
	segText   segmentKind = iota // template text between tags
	segOutput                    // {{ expression }}
	segBlock                     // {% statement %}
)

// A segment is one stretch of the template: text, or the tokens of one tag.
type segment struct {
	kind   segmentKind
	text   string
	tokens []token
	line   int
}

type tokenKind int

const (
	tokEnd tokenKind = iota // past the last token of a tag
	tokName
	tokString // text holds the decoded value
	tokInteger
	tokOperator
)

type token struct {
	kind tokenKind
	text string
	line int
}

// operators lists the operator tokens, longest first so that "==" is not read
// as two "=".
var operators = []string{
	"//", "**", "==", "!=", ">=", "<=",
	"+", "-", "/", "*", "%", "~", "[", "]", "(", ")", "{", "}", ">", "<", "=", ".", ":", "|", ",", ";",
}

type lexer struct {
	source string
	pos    int
	line   int
	// lineStart is true when the previous tag consumed the newline before
	// pos, or at the start of the template: lstrip_blocks then applies to a
	// tag even though no newline precedes it in the pending text.
	lineStart bool
	segments  []segment
}

// lex splits template source into text and tags, applying whitespace control.
func lex(source string) ([]segment, error) {
	l := &lexer{source: normalizeNewlines(source), line: 1, lineStart: true}
	for l.pos < len(l.source) {
		start := l.nextTag()
		if start < 0 {
			l.emitText(l.source[l.pos:])
			break
		}
		opener := l.source[start+1]
		after := start + 2
		var sign byte
		if after < len(l.source) && (l.source[after] == '-' || l.source[after] == '+') {
			sign = l.source[after]
			after++
		}

		text := l.source[l.pos:start]
		kept := text
		switch {
		case sign == '-':
			kept = strings.TrimRightFunc(text, isPythonSpace)
		case sign != '+' && opener != '{':
			kept = l.stripIndent(text)
		}
		l.emitText(kept)
		l.line += strings.Count(text[len(kept):], "\n")
		l.pos = after

		var err error
		if opener == '#' {
			err = l.skipComment()
		} else {
			err = l.lexTag(opener)
		}
		if err != nil {
			return nil, err
		}
	}
	return l.segments, nil
}

// normalizeNewlines turns every line ending into "\n" and drops one trailing
// newline.
func normalizeNewlines(source string) string {
	if strings.Contains(source, "\r") {
		source = strings.ReplaceAll(source, "\r\n", "\n")
		source = strings.ReplaceAll(source, "\r", "\n")
	}
	return strings.TrimSuffix(source, "\n")
}

// nextTag returns the offset of the next tag opener at or after pos, or -1.
func (l *lexer) nextTag() int {
	for i := l.pos; i+1 < len(l.source); i++ {
		if l.source[i] == '{' && strings.IndexByte("{%#", l.source[i+1]) >= 0 {
			return i
		}
	}
	return -1
}

func (l *lexer) emitText(text string) {
	if text != "" {
		l.segments = append(l.segments, segment{kind: segText, text: text, line: l.line})
		l.line += strings.Count(text, "\n")
	}
}

// stripIndent implements lstrip_blocks: whitespace between the start of a line
// and a block or comment tag is removed.
func (l *lexer) stripIndent(text string) string {
	lineStart := strings.LastIndexByte(text, '\n') + 1
	if lineStart == 0 && !l.lineStart {
		return text
	}
	if strings.TrimLeftFunc(text[lineStart:], isPythonSpace) != "" {
		return text
	}
	return text[:lineStart]
}

// skipComment consumes a comment body and its closing delimiter.
func (l *lexer) skipComment() error {
	end := strings.Index(l.source[l.pos:], "#}")
	if end < 0 {
		return fmt.Errorf("line %d: missing end of comment tag", l.line)
	}
	end += l.pos
	var sign byte
	if end > l.pos && (l.source[end-1] == '-' || l.source[end-1] == '+') {
		sign = l.source[end-1]
	}
	l.line += strings.Count(l.source[l.pos:end], "\n")
	l.finishTag(end+2, sign, true)
	return nil
}

// finishTag moves past a closing delimiter that ended at the given offset.
// A minus sign before it strips all following whitespace; otherwise
// trim_blocks removes one newline after a block or comment tag, unless a plus
// sign asks to keep it.
func (l *lexer) finishTag(end int, sign byte, trimBlock bool) {
	rest := l.source[end:]
	consumed := 0
	switch {
	case sign == '-':
		consumed = len(rest) - len(strings.TrimLeftFunc(rest, isPythonSpace))
	case sign != '+' && trimBlock && strings.HasPrefix(rest, "\n"):
		consumed = 1
	}
	l.line += strings.Count(rest[:consumed], "\n")
	l.pos = end + consumed
	l.lineStart = consumed > 0 && l.source[l.pos-1] == '\n'
}

// lexTag tokenizes one {{ }} or {% %} tag, whose opener has been consumed.
func (l *lexer) lexTag(opener byte) error {
	seg := segment{kind: segOutput, line: l.line}
	closer := "}}"
	if opener == '%' {
		seg.kind = segBlock
		closer = "%}"
	}
	var brackets []byte // closers the open brackets are waiting for

	for {
		rest := l.source[l.pos:]
		if rest == "" {
			return fmt.Errorf("line %d: unexpected end of template inside a tag", seg.line)
		}
		if len(brackets) == 0 {
			if end, sign, ok := tagEnd(rest, closer, opener == '%'); ok {
				l.segments = append(l.segments, seg)
				l.finishTag(l.pos+end, sign, opener == '%')
				return nil
			}
		}

		char, size := utf8.DecodeRuneInString(rest)
		first := rest[0]
		switch {
		case isPythonSpace(char):
			if char == '\n' {
				l.line++
			}
			l.pos += size
		case first >= '0' && first <= '9':
			text, err := l.lexInteger(rest)
			if err != nil {
				return err
			}
			seg.tokens = append(seg.tokens, token{tokInteger, text, l.line})
		case isNameStart(first):
			end := 1
			for end < len(rest) && isNamePart(rest[end]) {
				end++
			}
			seg.tokens = append(seg.tokens, token{tokName, rest[:end], l.line})
			l.pos += end
		case first == '\'' || first == '"':
			line := l.line
			value, err := l.lexString(rest)
			if err != nil {
				return err
			}
			seg.tokens = append(seg.tokens, token{tokString, value, line})
		default:
			op := matchOperator(rest)
			if op == "" {
				return fmt.Errorf("line %d: unexpected character %q in a tag", l.line, char)
			}
			switch op {
			case "(":
				brackets = append(brackets, ')')
			case "[":
				brackets = append(brackets, ']')
			case "{":
				brackets = append(brackets, '}')
			case ")", "]", "}":
				if len(brackets) == 0 || brackets[len(brackets)-1] != op[0] {
					return fmt.Errorf("line %d: unexpected %q", l.line, op)
				}
				brackets = brackets[:len(brackets)-1]
			}
			seg.tokens = append(seg.tokens, token{tokOperator, op, l.line})
			l.pos += len(op)
		}
	}
}

// tagEnd reports whether rest starts with the tag's closing delimiter,
// possibly preceded by a whitespace-control sign, and where the delimiter
// ends. Only block tags take a plus sign.
func tagEnd(rest, closer string, plusAllowed bool) (end int, sign byte, ok bool) {
	if strings.HasPrefix(rest, closer) {
		return len(closer), 0, true
	}
	if len(rest) > 0 && (rest[0] == '-' || rest[0] == '+' && plusAllowed) && strings.HasPrefix(rest[1:], closer) {
		return 1 + len(closer), rest[0], true
	}
	return 0, 0, false
}

func (l *lexer) lexInteger(rest string) (string, error) {
	end := 1
	for end < len(rest) && rest[end] >= '0' && rest[end] <= '9' {
		end++
	}
	switch {
	case end+1 < len(rest) && rest[end] == '.' && rest[end+1] >= '0' && rest[end+1] <= '9':
		return "", fmt.Errorf("line %d: float literals are not supported", l.line)
	case end < len(rest) && (rest[end] == '_' || isNameStart(rest[end])):
		return "", fmt.Errorf("line %d: unsupported number literal %q", l.line, rest[:end+1])
	case end > 1 && rest[0] == '0':
		return "", fmt.Errorf("line %d: invalid number literal %q", l.line, rest[:end])
	}
	l.pos += end
	return rest[:end], nil
}

// lexString decodes a quoted string literal starting at rest.
func (l *lexer) lexString(rest string) (string, error) {
	quote := rest[0]
	var value strings.Builder
	for i := 1; i < len(rest); i++ {
		char := rest[i]
		switch {
		case char == quote:
			l.pos += i + 1
			return value.String(), nil
		case char == '\\' && i+1 < len(rest):
			i++
			escaped, ok := stringEscapes[rest[i]]
			if !ok {
				return "", fmt.Errorf("line %d: unsupported string escape \\%c", l.line, []rune(rest[i:])[0])
			}
			value.WriteByte(escaped)
		default:
			if char == '\n' {
				l.line++
			}
			value.WriteByte(char)
		}
	}
	return "", fmt.Errorf("line %d: unterminated string literal", l.line)
}

// stringEscapes are the backslash escapes understood in string literals.
var stringEscapes = map[byte]byte{'n': '\n', 't': '\t', 'r': '\r', '\\': '\\', '\'': '\'', '"': '"'}

func matchOperator(rest string) string {
	for _, op := range operators {
		if strings.HasPrefix(rest, op) {
			return op
		}
	}
	return ""
}

func isNameStart(char byte) bool {
	return char == '_' || char >= 'a' && char <= 'z' || char >= 'A' && char <= 'Z'
}

func isNamePart(char byte) bool {
	return isNameStart(char) || char >= '0' && char <= '9'
}
