package chattemplate

import "strings"

// An expr is a parsed expression. Evaluating it yields a template value (see
// values.go) or the error Python would raise.
type expr interface {
	eval(sc *scope) (any, error)
}

// literalExpr is a constant: a string, an integer, a boolean or none.
type literalExpr struct{ value any }

func (e literalExpr) eval(*scope) (any, error) { return e.value, nil }

// nameExpr is a variable reference.
type nameExpr string

func (e nameExpr) eval(sc *scope) (any, error) { return sc.lookup(string(e)), nil }

// attrExpr is object.name.
type attrExpr struct {
	object expr
	name   string
}

func (e *attrExpr) eval(sc *scope) (any, error) {
	object, err := e.object.eval(sc)
	if err != nil {
		return nil, err
	}
	return getAttr(object, e.name)
}

// itemExpr is object[index].
type itemExpr struct {
	object expr
	index  expr
}

func (e *itemExpr) eval(sc *scope) (any, error) {
	object, err := e.object.eval(sc)
	if err != nil {
		return nil, err
	}
	index, err := e.index.eval(sc)
	if err != nil {
		return nil, err
	}
	return getItem(object, index)
}

// sliceExpr is object[start:stop:step]; an omitted part is nil.
type sliceExpr struct {
	object            expr
	start, stop, step expr
}

func (e *sliceExpr) eval(sc *scope) (any, error) {
	object, err := e.object.eval(sc)
	if err != nil {
		return nil, err
	}
	var bounds [3]any
	for i, part := range [...]expr{e.start, e.stop, e.step} {
		if part == nil {
			continue
		}
		if bounds[i], err = part.eval(sc); err != nil {
			return nil, err
		}
	}
	return getSlice(object, bounds[0], bounds[1], bounds[2])
}

// callExpr is fn(args, name=value).
type callExpr struct {
	fn     expr
	args   []expr
	kwargs []namedExpr
}

func (e *callExpr) eval(sc *scope) (any, error) {
	fn, err := e.fn.eval(sc)
	if err != nil {
		return nil, err
	}
	target, ok := fn.(callable)
	if !ok {
		if missing, isUndefined := fn.(undefined); isUndefined {
			return nil, missing.err()
		}
		return nil, pyError("TypeError", "'%s' object is not callable", typeName(fn))
	}
	args := make([]any, len(e.args))
	for i, arg := range e.args {
		if args[i], err = arg.eval(sc); err != nil {
			return nil, err
		}
	}
	kwargs, err := evalNamed(e.kwargs, sc)
	if err != nil {
		return nil, err
	}
	return target.call(args, kwargs)
}

// namedValue is an evaluated keyword argument.
type namedValue struct {
	name  string
	value any
}

func evalNamed(named []namedExpr, sc *scope) ([]namedValue, error) {
	if len(named) == 0 {
		return nil, nil
	}
	values := make([]namedValue, len(named))
	for i, arg := range named {
		value, err := arg.value.eval(sc)
		if err != nil {
			return nil, err
		}
		values[i] = namedValue{arg.name, value}
	}
	return values, nil
}

// filterExpr is value|filter(options).
type filterExpr struct {
	apply   filterFunc
	value   expr
	options []namedExpr
}

func (e *filterExpr) eval(sc *scope) (any, error) {
	value, err := e.value.eval(sc)
	if err != nil {
		return nil, err
	}
	options, err := evalNamed(e.options, sc)
	if err != nil {
		return nil, err
	}
	return e.apply(value, options)
}

// testExpr is "value is test" or "value is not test".
type testExpr struct {
	test    testFunc
	value   expr
	negated bool
}

func (e *testExpr) eval(sc *scope) (any, error) {
	value, err := e.value.eval(sc)
	if err != nil {
		return nil, err
	}
	return e.test(value) != e.negated, nil
}

// negExpr is -operand.
type negExpr struct{ operand expr }

func (e *negExpr) eval(sc *scope) (any, error) {
	operand, err := e.operand.eval(sc)
	if err != nil {
		return nil, err
	}
	return negate(operand)
}

// notExpr is "not operand".
type notExpr struct{ operand expr }

func (e *notExpr) eval(sc *scope) (any, error) {
	operand, err := e.operand.eval(sc)
	if err != nil {
		return nil, err
	}
	return !truthy(operand), nil
}

// andExpr is "left and right": the left value if it is falsy, else the right.
type andExpr struct{ left, right expr }

func (e *andExpr) eval(sc *scope) (any, error) {
	left, err := e.left.eval(sc)
	if err != nil || !truthy(left) {
		return left, err
	}
	return e.right.eval(sc)
}

// orExpr is "left or right": the left value if it is truthy, else the right.
type orExpr struct{ left, right expr }

func (e *orExpr) eval(sc *scope) (any, error) {
	left, err := e.left.eval(sc)
	if err != nil || truthy(left) {
		return left, err
	}
	return e.right.eval(sc)
}

// compareExpr is "left op right" for ==, !=, <, <=, >, >=, in and not in.
type compareExpr struct {
	op          string
	left, right expr
}

func (e *compareExpr) eval(sc *scope) (any, error) {
	left, err := e.left.eval(sc)
	if err != nil {
		return nil, err
	}
	right, err := e.right.eval(sc)
	if err != nil {
		return nil, err
	}
	switch e.op {
	case "==":
		return equal(left, right), nil
	case "!=":
		return !equal(left, right), nil
	case "in":
		return contains(right, left)
	case "not in":
		found, err := contains(right, left)
		return !found, err
	}
	return order(e.op, left, right)
}

// arithmeticExpr is "left + right" or "left - right".
type arithmeticExpr struct {
	op          byte
	left, right expr
}

func (e *arithmeticExpr) eval(sc *scope) (any, error) {
	left, err := e.left.eval(sc)
	if err != nil {
		return nil, err
	}
	right, err := e.right.eval(sc)
	if err != nil {
		return nil, err
	}
	if e.op == '+' {
		return add(left, right)
	}
	return subtract(left, right)
}

// concatExpr is "a ~ b ~ c": the operands' text, joined.
type concatExpr struct{ operands []expr }

func (e *concatExpr) eval(sc *scope) (any, error) {
	var joined strings.Builder
	for _, operand := range e.operands {
		value, err := operand.eval(sc)
		if err != nil {
			return nil, err
		}
		text, err := pyStr(value)
		if err != nil {
			return nil, err
		}
		joined.WriteString(text)
	}
	return joined.String(), nil
}

// condExpr is "then if condition else otherwise". Without an else part it is
// undefined when the condition is false.
type condExpr struct {
	condition expr
	then      expr
	otherwise expr
}

func (e *condExpr) eval(sc *scope) (any, error) {
	condition, err := e.condition.eval(sc)
	if err != nil {
		return nil, err
	}
	switch {
	case truthy(condition):
		return e.then.eval(sc)
	case e.otherwise != nil:
		return e.otherwise.eval(sc)
	}
	return undefined{hint: "the inline if-expression evaluated to false and no else section was defined"}, nil
}
