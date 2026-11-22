package chattemplate

import "strings"

// A statement is one piece of a template body: literal text, an output
// expression or a block tag with its own body.
type statement interface {
	exec(out *strings.Builder, sc *scope) error
}

func execAll(body []statement, out *strings.Builder, sc *scope) error {
	for _, stmt := range body {
		if err := stmt.exec(out, sc); err != nil {
			return err
		}
	}
	return nil
}

// textStmt is template text outside any tag.
type textStmt string

func (s textStmt) exec(out *strings.Builder, _ *scope) error {
	out.WriteString(string(s))
	return nil
}

// outputStmt is {{ value }}.
type outputStmt struct{ value expr }

func (s *outputStmt) exec(out *strings.Builder, sc *scope) error {
	value, err := s.value.eval(sc)
	if err != nil {
		return err
	}
	text, err := pyStr(value)
	if err != nil {
		return err
	}
	out.WriteString(text)
	return nil
}

// ifStmt is {% if %} with its {% elif %} branches and optional {% else %}.
// It does not open a scope: a variable set inside is visible after it.
type ifStmt struct {
	branches  []ifBranch
	otherwise []statement
}

type ifBranch struct {
	condition expr
	body      []statement
}

func (s *ifStmt) exec(out *strings.Builder, sc *scope) error {
	for _, branch := range s.branches {
		condition, err := branch.condition.eval(sc)
		if err != nil {
			return err
		}
		if truthy(condition) {
			return execAll(branch.body, out, sc)
		}
	}
	return execAll(s.otherwise, out, sc)
}

// forStmt is {% for targets in iterable %}. Each pass runs the body in a
// scope of its own, so a variable set in the body is gone on the next pass
// and after the loop.
type forStmt struct {
	targets   []string
	iterable  expr
	predicate expr
	body      []statement
	otherwise []statement
}

func (s *forStmt) exec(out *strings.Builder, sc *scope) error {
	iterable, err := s.iterable.eval(sc)
	if err != nil {
		return err
	}
	items, err := iterate(iterable)
	if err != nil {
		return err
	}
	if s.predicate != nil {
		return s.execFiltered(out, sc, items)
	}
	if len(items) == 0 {
		return execAll(s.otherwise, out, sc.child())
	}
	loop := &loopState{items: items}
	for index, item := range items {
		pass := sc.child()
		if err := bindTargets(pass, s.targets, item); err != nil {
			return err
		}
		loop.index0 = index
		pass.set("loop", loop)
		if err := execAll(s.body, out, pass); err != nil {
			return err
		}
	}
	return nil
}

// loopState is the "loop" variable of a for body.
type loopState struct {
	items  []any
	filter *filteredItems
	index0 int
}

// attr returns the loop attribute with the given name, as in
// jinja2.runtime.LoopContext.
func (l *loopState) attr(name string) (any, bool) {
	switch name {
	case "index":
		return l.index0 + 1, true
	case "index0":
		return l.index0, true
	case "revindex":
		return len(l.items) - l.index0, true
	case "revindex0":
		return len(l.items) - l.index0 - 1, true
	case "first":
		return l.index0 == 0, true
	case "last":
		return l.index0 == len(l.items)-1, true
	case "length":
		return len(l.items), true
	case "depth":
		return 1, true // deeper only in a recursive loop, which is not supported
	case "depth0":
		return 0, true
	case "previtem":
		if l.index0 == 0 {
			return undefined{hint: "there is no previous item"}, true
		}
		return l.items[l.index0-1], true
	case "nextitem":
		if l.index0 == len(l.items)-1 {
			return undefined{hint: "there is no next item"}, true
		}
		return l.items[l.index0+1], true
	}
	return nil, false
}

// setStmt is {% set name = value %} or, when attr is not empty,
// {% set name.attr = value %}.
type setStmt struct {
	name  string
	attr  string
	value expr
}

func (s *setStmt) exec(_ *strings.Builder, sc *scope) error {
	value, err := s.value.eval(sc)
	if err != nil {
		return err
	}
	if s.attr == "" {
		sc.set(s.name, value)
		return nil
	}
	target, ok := sc.lookup(s.name).(*namespace)
	if !ok {
		return pyError("TemplateRuntimeError", "cannot assign attribute on non-namespace object")
	}
	target.attrs[s.attr] = value
	return nil
}

// macroStmt is {% macro name(params) %}: it binds name to the macro.
type macroStmt struct {
	name   string
	params []macroParam
	body   []statement
}

func (s *macroStmt) exec(_ *strings.Builder, sc *scope) error {
	sc.set(s.name, &macro{name: s.name, params: s.params, body: s.body, root: sc.root})
	return nil
}
