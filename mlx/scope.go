package mlx

// scope owns the arrays created while it is the innermost open scope.
type scope struct {
	parent *scope
	head   *Array // newest owned array; the rest follow through Array.next
}

// current is the innermost open scope, or nil outside any Scoped call. A plain
// global, because the package is single-threaded by contract.
var current *scope

// Scoped runs fn and returns its error. Every Array created inside fn is freed
// when fn returns or panics, except those passed to Keep. An operation that
// fails inside fn panics with *Error; Scoped returns that error instead. Any
// other panic passes through.
func Scoped(fn func() error) (err error) {
	s := &scope{parent: current}
	current = s
	defer func() {
		current = s.parent
		s.close()
		switch r := recover().(type) {
		case nil:
		case *Error:
			err = r
		default:
			panic(r)
		}
	}()
	return fn()
}

// Keep moves arrays owned by the innermost scope to the enclosing scope. At
// the outermost scope the caller becomes the owner and must Free them. Arrays
// the innermost scope does not own are left alone.
func Keep(arrs ...*Array) {
	if current == nil {
		return
	}
	for _, a := range arrs {
		if a.owner == current {
			a.owner = current.parent
		}
	}
}

// adopt makes s the owner of a.
func (s *scope) adopt(a *Array) {
	a.owner = s
	a.next = s.head
	s.head = a
}

// close frees the arrays s still owns and hands the kept ones to their new
// owner. A kept array without an owner belongs to the caller.
func (s *scope) close() {
	for a := s.head; a != nil; {
		next := a.next
		a.next = nil
		switch a.owner {
		case s:
			a.Free()
		case nil:
		default:
			a.owner.adopt(a)
		}
		a = next
	}
	s.head = nil
}
