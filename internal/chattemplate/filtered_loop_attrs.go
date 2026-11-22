package chattemplate

// filteredAttr resolves loop attributes against the filtered item sequence.
func (l *loopState) filteredAttr(name string) (any, bool, error) {
	switch name {
	case "last", "nextitem":
		v, ok, err := l.filter.at(l.index0 + 1)
		if err != nil {
			return nil, false, err
		}
		if name == "last" {
			return !ok, true, nil
		}
		if !ok {
			return undefined{hint: "there is no next item"}, true, nil
		}
		return v, true, nil
	case "previtem":
		if l.index0 > 0 {
			return l.filter.at(l.index0 - 1)
		}
		return undefined{hint: "there is no previous item"}, true, nil
	case "length":
		n, err := l.filter.length()
		if err != nil {
			return nil, false, err
		}
		return n, true, nil
	case "revindex", "revindex0":
		n, err := l.filter.length()
		if err != nil {
			return nil, false, err
		}
		if name == "revindex0" {
			return n - l.index0 - 1, true, nil
		}
		return n - l.index0, true, nil
	default:
		v, ok := l.attr(name)
		return v, ok, nil
	}
}
