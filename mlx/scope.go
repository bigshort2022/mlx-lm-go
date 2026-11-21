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
//
// A Scoped call that is not nested in another is the outermost scope. If it
// fails, that is if fn returns an error or panics, it also frees the arrays
// that were kept for the caller, who gets no chance to take them over.
func Scoped(fn func() error) (err error) {
	s := &scope{parent: current}
	current = s
	defer func() {
		r := recover()
		current = s.parent
		s.close(r != nil || err != nil)
		switch r := r.(type) {
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
// the outermost scope the caller becomes the owner once Scoped returns
// without error, and must Free them; if the outermost scope fails, Scoped
// frees them instead. Nil arrays and arrays the innermost scope does not own
// are left alone.
func Keep(arrs ...*Array) {
	if current == nil {
		return
	}
	for _, a := range arrs {
		if a != nil && a.owner == current {
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
// owner. A kept array without an owner is one the outermost scope kept for
// the caller: it stays alive unless the scope failed.
func (s *scope) close(failed bool) {
	for a := s.head; a != nil; {
		next := a.next
		a.next = nil
		switch a.owner {
		case s:
			a.Free()
		case nil:
			if failed {
				a.Free()
			}
		default:
			a.owner.adopt(a)
		}
		a = next
	}
	s.head = nil
}
