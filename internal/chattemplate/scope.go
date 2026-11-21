package chattemplate

// scope holds the variables visible at one point of a render. The root scope
// holds the template's variables; a for body and a macro call each get a
// child scope, so what they set does not leak out.
type scope struct {
	vars   map[string]any
	parent *scope
	root   *scope
}

func newRootScope(vars map[string]any) *scope {
	root := &scope{vars: vars}
	root.root = root
	return root
}

func (s *scope) child() *scope {
	return &scope{parent: s, root: s.root}
}

// lookup finds a variable in this scope or the ones around it, then among the
// global functions. A name that is not bound anywhere is undefined.
func (s *scope) lookup(name string) any {
	for at := s; at != nil; at = at.parent {
		if value, ok := at.vars[name]; ok {
			return value
		}
	}
	if global, ok := globals[name]; ok {
		return global
	}
	return undefined{hint: "'" + name + "' is undefined"}
}

func (s *scope) set(name string, value any) {
	if s.vars == nil {
		s.vars = map[string]any{}
	}
	s.vars[name] = value
}
