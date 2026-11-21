package chattemplate

import (
	"fmt"
	"slices"
)

// parser turns the lexer's segments into statements. The grammar follows
// jinja2/parser.py for the constructs it supports and rejects the rest by
// name.
type parser struct {
	segments []segment
	pos      int
	// forDepth counts the for bodies around the statement being parsed, and
	// macro is the macro whose body it belongs to, if any.
	forDepth int
	macro    *macroStmt
}

func (p *parser) newTag(seg segment) *tagReader {
	return &tagReader{tokens: seg.tokens, line: seg.line, macro: p.macro}
}

// parse turns template source into the statements of its body.
func parse(source string) ([]statement, error) {
	segments, err := lex(source)
	if err != nil {
		return nil, err
	}
	p := &parser{segments: segments}
	body, _, err := p.parseBody()
	return body, err
}

// parseBody parses statements up to a block tag whose keyword is one of the
// terminators, and returns that tag with the keyword consumed. The last
// terminator is the one that closes the block. Without terminators it parses
// to the end of the template.
func (p *parser) parseBody(terminators ...string) ([]statement, *tagReader, error) {
	var body []statement
	for p.pos < len(p.segments) {
		seg := p.segments[p.pos]
		p.pos++
		switch seg.kind {
		case segText:
			body = append(body, textStmt(seg.text))
		case segOutput:
			tag := p.newTag(seg)
			value, err := tag.parseExpression()
			if err != nil {
				return nil, nil, err
			}
			if err := tag.expectEnd(); err != nil {
				return nil, nil, err
			}
			body = append(body, &outputStmt{value: value})
		case segBlock:
			tag := p.newTag(seg)
			keyword := tag.next()
			if keyword.kind != tokName {
				return nil, nil, tag.errorf("expected a statement name")
			}
			if slices.Contains(terminators, keyword.text) {
				return body, tag, nil
			}
			stmt, err := p.parseStatement(keyword.text, tag)
			if err != nil {
				return nil, nil, err
			}
			body = append(body, stmt)
		}
	}
	if len(terminators) > 0 {
		return nil, nil, fmt.Errorf("unexpected end of template: missing {%% %s %%}", terminators[len(terminators)-1])
	}
	return body, nil, nil
}

// parseStatement parses the block statement that starts with keyword.
func (p *parser) parseStatement(keyword string, tag *tagReader) (statement, error) {
	switch keyword {
	case "if":
		return p.parseIf(tag)
	case "for":
		return p.parseFor(tag)
	case "set":
		return parseSet(tag)
	case "macro":
		return p.parseMacro(tag)
	case "elif", "else", "endif", "endfor", "endmacro":
		return nil, tag.errorf("unexpected {%% %s %%}", keyword)
	}
	return nil, tag.errorf("statement %q is not supported", keyword)
}

// parseIf parses {% if %} with its {% elif %} and {% else %} branches.
// As in Jinja, a condition is not a full expression: "a if b else c" needs
// parentheses there.
func (p *parser) parseIf(tag *tagReader) (statement, error) {
	stmt := &ifStmt{}
	for {
		condition, err := tag.parseOr()
		if err != nil {
			return nil, err
		}
		if err := tag.expectEnd(); err != nil {
			return nil, err
		}
		body, next, err := p.parseBody("elif", "else", "endif")
		if err != nil {
			return nil, err
		}
		stmt.branches = append(stmt.branches, ifBranch{condition, body})

		switch next.keyword() {
		case "elif":
			tag = next
			continue
		case "else":
			if err := next.expectEnd(); err != nil {
				return nil, err
			}
			if stmt.otherwise, next, err = p.parseBody("endif"); err != nil {
				return nil, err
			}
		}
		return stmt, next.expectEnd()
	}
}

// expectTarget reads a name that a statement assigns to.
func (r *tagReader) expectTarget() (string, error) {
	name := r.peek()
	if name.kind != tokName {
		return "", r.errorf("expected a name, found %s", name.describe())
	}
	if _, reserved := constants[name.text]; reserved {
		return "", r.errorf("cannot assign to %q", name.text)
	}
	r.pos++
	return name.text, nil
}

// parseFor parses {% for a, b in iterable %} ... {% endfor %}.
func (p *parser) parseFor(tag *tagReader) (statement, error) {
	stmt := &forStmt{}
	for {
		target, err := tag.expectTarget()
		if err != nil {
			return nil, err
		}
		stmt.targets = append(stmt.targets, target)
		if !tag.acceptOp(",") {
			break
		}
	}
	if !tag.acceptName("in") {
		return nil, tag.errorf("expected \"in\", found %s", tag.peek().describe())
	}
	var err error
	if stmt.iterable, err = tag.parseOr(); err != nil {
		return nil, err
	}
	for _, modifier := range []string{"if", "recursive"} {
		if tag.isName(modifier) {
			return nil, tag.errorf("for ... %s is not supported", modifier)
		}
	}
	if err := tag.expectEnd(); err != nil {
		return nil, err
	}

	p.forDepth++
	body, end, err := p.parseBody("else", "endfor")
	p.forDepth--
	if err != nil {
		return nil, err
	}
	if end.keyword() == "else" {
		return nil, end.errorf("for ... else is not supported")
	}
	stmt.body = body
	return stmt, end.expectEnd()
}

// parseSet parses {% set name = value %} and {% set name.attr = value %}.
func parseSet(tag *tagReader) (statement, error) {
	stmt := &setStmt{}
	var err error
	if stmt.name, err = tag.expectTarget(); err != nil {
		return nil, err
	}
	if tag.acceptOp(".") {
		attr := tag.next()
		if attr.kind != tokName {
			tag.pos--
			return nil, tag.errorf("expected an attribute name, found %s", attr.describe())
		}
		stmt.attr = attr.text
	}
	switch {
	case tag.isOp(","):
		return nil, tag.errorf("set with several targets is not supported")
	case !tag.acceptOp("="):
		return nil, tag.errorf("block set ({%% set name %%}...{%% endset %%}) is not supported")
	}
	if stmt.value, err = tag.parseExpression(); err != nil {
		return nil, err
	}
	return stmt, tag.expectEnd()
}

// parseMacro parses {% macro name(a, b=default) %} ... {% endmacro %}.
func (p *parser) parseMacro(tag *tagReader) (statement, error) {
	// A macro body here sees the template's top-level variables. In Jinja it
	// also sees the variables around its definition, so one defined inside a
	// for body or another macro would behave differently.
	switch {
	case p.macro != nil:
		return nil, tag.errorf("macro inside a macro is not supported")
	case p.forDepth > 0:
		return nil, tag.errorf("macro inside a for loop is not supported")
	}
	stmt := &macroStmt{}
	var err error
	if stmt.name, err = tag.expectTarget(); err != nil {
		return nil, err
	}
	if err := tag.expectOp("("); err != nil {
		return nil, err
	}
	for !tag.isOp(")") {
		if len(stmt.params) > 0 {
			if err := tag.expectOp(","); err != nil {
				return nil, err
			}
		}
		param := macroParam{}
		if tag.isName("caller") {
			return nil, tag.errorf("macro parameter %q is not supported", "caller")
		}
		if param.name, err = tag.expectTarget(); err != nil {
			return nil, err
		}
		switch {
		case tag.acceptOp("="):
			if param.fallback, err = tag.parseExpression(); err != nil {
				return nil, err
			}
		case len(stmt.params) > 0 && stmt.params[len(stmt.params)-1].fallback != nil:
			return nil, tag.errorf("non-default argument follows default argument")
		}
		stmt.params = append(stmt.params, param)
	}
	if err := tag.expectOp(")"); err != nil {
		return nil, err
	}
	if err := tag.expectEnd(); err != nil {
		return nil, err
	}

	p.macro = stmt
	body, end, err := p.parseBody("endmacro")
	p.macro = nil
	if err != nil {
		return nil, err
	}
	stmt.body = body
	return stmt, end.expectEnd()
}
