package chattemplate

import "strings"

func bindTargets(pass *scope, targets []string, item any) error {
	if len(targets) == 1 {
		pass.set(targets[0], item)
		return nil
	}
	vals, err := unpack(item, len(targets))
	if err != nil {
		return err
	}
	for i, t := range targets {
		pass.set(t, vals[i])
	}
	return nil
}

func (s *forStmt) execFiltered(out *strings.Builder, sc *scope, items []any) error {
	f := &filteredItems{
		source: items,
		accept: func(item any) (any, bool, error) {
			if len(s.targets) > 1 {
				values, err := unpack(item, len(s.targets))
				if err != nil {
					return nil, false, err
				}
				item = tuple(values)
			}
			scope := sc.child()
			if err := bindTargets(scope, s.targets, item); err != nil {
				return nil, false, err
			}
			v, err := s.predicate.eval(scope)
			if err != nil {
				return nil, false, err
			}
			return item, truthy(v), nil
		},
	}
	l := &loopState{filter: f}
	for i := 0; ; i++ {
		item, ok, err := f.at(i)
		if err != nil {
			return err
		}
		if !ok {
			if i == 0 {
				return execAll(s.otherwise, out, sc.child())
			}
			return nil
		}
		pass := sc.child()
		if err := bindTargets(pass, s.targets, item); err != nil {
			return err
		}
		l.index0 = i
		pass.set("loop", l)
		if err := execAll(s.body, out, pass); err != nil {
			return err
		}
	}
}
