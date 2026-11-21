package chattemplate

import (
	"slices"
	"strconv"
)

// The methods below parse expressions with the precedence of
// jinja2/parser.py, loosest binding first.

// parseExpression parses a full expression, including "a if b else c".
func (r *tagReader) parseExpression() (expr, error) {
	value, err := r.parseOr()
	if err != nil {
		return nil, err
	}
	for r.acceptName("if") {
		condition, err := r.parseOr()
		if err != nil {
			return nil, err
		}
		var otherwise expr
		if r.acceptName("else") {
			if otherwise, err = r.parseExpression(); err != nil {
				return nil, err
			}
		}
		value = &condExpr{condition: condition, then: value, otherwise: otherwise}
	}
	return value, nil
}

func (r *tagReader) parseOr() (expr, error) {
	left, err := r.parseAnd()
	for err == nil && r.acceptName("or") {
		var right expr
		if right, err = r.parseAnd(); err == nil {
			left = &orExpr{left, right}
		}
	}
	return left, err
}

func (r *tagReader) parseAnd() (expr, error) {
	left, err := r.parseNot()
	for err == nil && r.acceptName("and") {
		var right expr
		if right, err = r.parseNot(); err == nil {
			left = &andExpr{left, right}
		}
	}
	return left, err
}

func (r *tagReader) parseNot() (expr, error) {
	if r.acceptName("not") {
		operand, err := r.parseNot()
		if err != nil {
			return nil, err
		}
		return &notExpr{operand}, nil
	}
	return r.parseCompare()
}

// comparisonOps are the operator tokens parseCompare accepts; "in" and
// "not in" are names and handled there.
var comparisonOps = []string{"==", "!=", "<=", ">=", "<", ">"}

// acceptComparison consumes a comparison operator and returns it, or "".
func (r *tagReader) acceptComparison() string {
	for _, op := range comparisonOps {
		if r.acceptOp(op) {
			return op
		}
	}
	if r.acceptName("in") {
		return "in"
	}
	if r.isName("not") && r.followedBy(tokName, "in") {
		r.pos += 2
		return "not in"
	}
	return ""
}

func (r *tagReader) parseCompare() (expr, error) {
	left, err := r.parseAdditive()
	if err != nil {
		return nil, err
	}
	op := r.acceptComparison()
	if op == "" {
		return left, nil
	}
	right, err := r.parseAdditive()
	if err != nil {
		return nil, err
	}
	if next := r.acceptComparison(); next != "" {
		r.pos--
		return nil, r.errorf("chained comparisons are not supported")
	}
	return &compareExpr{op: op, left: left, right: right}, nil
}

// parseAdditive parses + and -, which bind looser than ~.
func (r *tagReader) parseAdditive() (expr, error) {
	left, err := r.parseConcat()
	for err == nil {
		var op byte
		switch {
		case r.acceptOp("+"):
			op = '+'
		case r.acceptOp("-"):
			op = '-'
		default:
			return left, nil
		}
		var right expr
		if right, err = r.parseConcat(); err == nil {
			left = &arithmeticExpr{op: op, left: left, right: right}
		}
	}
	return nil, err
}

func (r *tagReader) parseConcat() (expr, error) {
	first, err := r.parseMultiplicative()
	if err != nil || !r.isOp("~") {
		return first, err
	}
	operands := []expr{first}
	for r.acceptOp("~") {
		operand, err := r.parseMultiplicative()
		if err != nil {
			return nil, err
		}
		operands = append(operands, operand)
	}
	return &concatExpr{operands}, nil
}

// parseMultiplicative stands where Jinja parses *, /, //, % and **. No chat
// template this package supports uses them.
func (r *tagReader) parseMultiplicative() (expr, error) {
	value, err := r.parseUnary(true)
	if err != nil {
		return nil, err
	}
	for _, op := range []string{"*", "/", "//", "%", "**"} {
		if r.isOp(op) {
			return nil, r.errorf("operator %q is not supported", op)
		}
	}
	return value, nil
}

// parseUnary parses a primary with its postfix operators. Filters and tests
// apply to the whole of "-x", so they are parsed only at the outermost level.
func (r *tagReader) parseUnary(withFilters bool) (expr, error) {
	var value expr
	var err error
	switch {
	case r.acceptOp("-"):
		var operand expr
		if operand, err = r.parseUnary(false); err == nil {
			value = &negExpr{operand}
		}
	case r.isOp("+"):
		return nil, r.errorf("unary %q is not supported", "+")
	default:
		value, err = r.parsePrimary()
	}
	if err == nil {
		value, err = r.parsePostfix(value)
	}
	if err == nil && withFilters {
		value, err = r.parseFilters(value)
	}
	return value, err
}

// constants are the names that denote a literal value.
var constants = map[string]any{
	"true": true, "True": true,
	"false": false, "False": false,
	"none": nil, "None": nil,
}

func (r *tagReader) parsePrimary() (expr, error) {
	tok := r.peek()
	switch {
	case tok.kind == tokString:
		r.pos++
		return literalExpr{tok.text}, nil
	case tok.kind == tokInteger:
		value, err := strconv.Atoi(tok.text)
		if err != nil {
			return nil, r.errorf("integer literal %s is out of range", tok.text)
		}
		r.pos++
		return literalExpr{value}, nil
	case tok.kind == tokName:
		r.pos++
		if value, ok := constants[tok.text]; ok {
			return literalExpr{value}, nil
		}
		return nameExpr(tok.text), nil
	case r.acceptOp("("):
		value, err := r.parseExpression()
		if err != nil {
			return nil, err
		}
		if r.isOp(",") {
			return nil, r.errorf("tuple literals are not supported")
		}
		return value, r.expectOp(")")
	case r.isOp("["):
		return nil, r.errorf("list literals are not supported")
	case r.isOp("{"):
		return nil, r.errorf("dict literals are not supported")
	}
	return nil, r.errorf("unexpected %s", tok.describe())
}

// parsePostfix parses attribute access, subscripts and calls after a value.
func (r *tagReader) parsePostfix(value expr) (expr, error) {
	for {
		switch {
		case r.acceptOp("."):
			name := r.next()
			if name.kind != tokName {
				r.pos--
				return nil, r.errorf("expected an attribute name, found %s", name.describe())
			}
			value = &attrExpr{object: value, name: name.text}
		case r.acceptOp("["):
			var err error
			if value, err = r.parseSubscript(value); err != nil {
				return nil, err
			}
		case r.isOp("("):
			var err error
			if value, err = r.parseCall(value); err != nil {
				return nil, err
			}
		default:
			return value, nil
		}
	}
}

// parseSubscript parses what follows "[": an index or a slice.
func (r *tagReader) parseSubscript(object expr) (expr, error) {
	var bounds [3]expr // start, stop, step
	part := 0
	isSlice := false
	for !r.isOp("]") {
		if r.acceptOp(":") {
			isSlice = true
			if part++; part == len(bounds) {
				return nil, r.errorf("a slice has at most three parts")
			}
			continue
		}
		if bounds[part] != nil {
			break // two expressions with no colon between them
		}
		var err error
		if bounds[part], err = r.parseExpression(); err != nil {
			return nil, err
		}
	}
	if err := r.expectOp("]"); err != nil {
		return nil, err
	}
	if isSlice {
		return &sliceExpr{object: object, start: bounds[0], stop: bounds[1], step: bounds[2]}, nil
	}
	if bounds[0] == nil {
		return nil, r.errorf("expected an index inside []")
	}
	return &itemExpr{object: object, index: bounds[0]}, nil
}

// namedExpr is a keyword argument as written: name=value.
type namedExpr struct {
	name  string
	value expr
}

// parseArguments parses a parenthesized argument list.
func (r *tagReader) parseArguments() (args []expr, kwargs []namedExpr, err error) {
	if err := r.expectOp("("); err != nil {
		return nil, nil, err
	}
	for !r.isOp(")") {
		if len(args)+len(kwargs) > 0 {
			if err := r.expectOp(","); err != nil {
				return nil, nil, err
			}
			if r.isOp(")") {
				break // trailing comma
			}
		}
		if r.isOp("*") || r.isOp("**") {
			return nil, nil, r.errorf("argument unpacking with %q is not supported", r.peek().text)
		}
		name := r.peek()
		if name.kind == tokName && r.followedBy(tokOperator, "=") {
			r.pos += 2
			value, err := r.parseExpression()
			if err != nil {
				return nil, nil, err
			}
			kwargs = append(kwargs, namedExpr{name.text, value})
			continue
		}
		if len(kwargs) > 0 {
			return nil, nil, r.errorf("positional argument follows keyword argument")
		}
		value, err := r.parseExpression()
		if err != nil {
			return nil, nil, err
		}
		args = append(args, value)
	}
	return args, kwargs, r.expectOp(")")
}

func (r *tagReader) parseCall(fn expr) (expr, error) {
	if name, ok := fn.(nameExpr); ok && slices.Contains(unsupportedGlobals, string(name)) {
		return nil, r.errorf("function %q is not supported", string(name))
	}
	args, kwargs, err := r.parseArguments()
	if err != nil {
		return nil, err
	}
	return &callExpr{fn: fn, args: args, kwargs: kwargs}, nil
}

// parseFilters parses the "| filter" and "is test" suffixes of a value.
func (r *tagReader) parseFilters(value expr) (expr, error) {
	for {
		var err error
		switch {
		case r.acceptOp("|"):
			value, err = r.parseFilter(value)
		case r.acceptName("is"):
			value, err = r.parseTest(value)
		case r.isOp("("):
			value, err = r.parseCall(value)
		default:
			return value, nil
		}
		if err != nil {
			return nil, err
		}
	}
}

func (r *tagReader) parseFilter(value expr) (expr, error) {
	name := r.next()
	spec, known := filters[name.text]
	if name.kind != tokName || !known {
		r.pos--
		return nil, r.errorf("filter %s is not supported", name.describe())
	}
	filtered := &filterExpr{apply: spec.apply, value: value}
	if !r.isOp("(") {
		return filtered, nil
	}
	args, kwargs, err := r.parseArguments()
	if err != nil {
		return nil, err
	}
	if len(args) > 0 {
		return nil, r.errorf("filter %q: positional arguments are not supported", name.text)
	}
	for _, option := range kwargs {
		if !spec.takes(option.name) {
			return nil, r.errorf("filter %q: argument %q is not supported", name.text, option.name)
		}
	}
	filtered.options = kwargs
	return filtered, nil
}

func (r *tagReader) parseTest(value expr) (expr, error) {
	negated := r.acceptName("not")
	name := r.next()
	test, known := tests[name.text]
	if name.kind != tokName || !known {
		r.pos--
		return nil, r.errorf("test %s is not supported", name.describe())
	}
	// Jinja reads a following value as the test's argument, unless it is one
	// of the words that continue the enclosing expression.
	next := r.peek()
	switch {
	case next.kind == tokString, next.kind == tokInteger, r.isOp("("), r.isOp("["), r.isOp("{"):
		return nil, r.errorf("test %q: arguments are not supported", name.text)
	case next.kind == tokName && next.text != "else" && next.text != "or" && next.text != "and":
		return nil, r.errorf("test %q: arguments are not supported", name.text)
	}
	return &testExpr{test: test, value: value, negated: negated}, nil
}
