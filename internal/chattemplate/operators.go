package chattemplate

import (
	"cmp"
	"errors"
	"math"
	"math/big"
	"slices"
	"strings"
)

// Python's operators on template values. Each raises what Python raises:
// computing with undefined is an UndefinedError, mixing types a TypeError.

// number is a Python number: int, float or bool, which counts as 0 or 1.
type number struct {
	integer int
	float   float64
	isFloat bool
}

func asNumber(value any) (number, bool) {
	switch value := value.(type) {
	case int:
		return number{integer: value}, true
	case float64:
		return number{float: value, isFloat: true}, true
	case bool:
		if value {
			return number{integer: 1}, true
		}
		return number{}, true
	}
	return number{}, false
}

func (n number) asFloat() float64 {
	if n.isFloat {
		return n.float
	}
	return float64(n.integer)
}

// asText returns the text of a string or safe string.
func asText(value any) (string, bool) {
	switch value := value.(type) {
	case string:
		return value, true
	case safeString:
		return string(value), true
	}
	return "", false
}

// firstUndefined returns the error for the first undefined operand, if any.
func firstUndefined(operands ...any) error {
	for _, operand := range operands {
		if missing, ok := operand.(undefined); ok {
			return missing.err()
		}
	}
	return nil
}

// Python integers grow without bound; ours fail instead of wrapping around.
var errIntegerRange = errors.New("integer result is out of range")

// An integer beyond 64 bits can be read from a request, compared for
// equality and rendered, which is all a chat template does with such a
// number; computing with one is not implemented.
var errBigArithmetic = errors.New("arithmetic on an integer beyond 64 bits is not supported")

func isBig(value any) bool {
	_, ok := value.(*big.Int)
	return ok
}

// equalsBig is integer == other, where integer does not fit an int.
func equalsBig(integer *big.Int, other any) bool {
	switch other := other.(type) {
	case *big.Int:
		return integer.Cmp(other) == 0
	case float64:
		if math.IsNaN(other) || math.IsInf(other, 0) {
			return false
		}
		whole, accuracy := big.NewFloat(other).Int(nil)
		return accuracy == big.Exact && integer.Cmp(whole) == 0
	}
	return false
}

// add is left + right. Python escapes the other operand when one is a safe
// string; no chat template relies on that, so here it is a type error like
// any other mix.
func add(left, right any) (any, error) {
	if err := firstUndefined(left, right); err != nil {
		return nil, err
	}
	if isBig(left) || isBig(right) {
		return nil, errBigArithmetic
	}
	if a, ok := asNumber(left); ok {
		if b, ok := asNumber(right); ok {
			if a.isFloat || b.isFloat {
				return a.asFloat() + b.asFloat(), nil
			}
			sum := a.integer + b.integer
			if (a.integer^sum)&(b.integer^sum) < 0 {
				return nil, errIntegerRange
			}
			return sum, nil
		}
	}
	switch left := left.(type) {
	case string:
		if right, ok := right.(string); ok {
			return left + right, nil
		}
		return nil, pyError("TypeError", `can only concatenate str (not "%s") to str`, typeName(right))
	case []any:
		if right, ok := right.([]any); ok {
			return slices.Concat(left, right), nil
		}
		return nil, pyError("TypeError", `can only concatenate list (not "%s") to list`, typeName(right))
	}
	return nil, pyError("TypeError", "unsupported operand type(s) for +: '%s' and '%s'", typeName(left), typeName(right))
}

// subtract is left - right.
func subtract(left, right any) (any, error) {
	if err := firstUndefined(left, right); err != nil {
		return nil, err
	}
	if isBig(left) || isBig(right) {
		return nil, errBigArithmetic
	}
	a, aOK := asNumber(left)
	b, bOK := asNumber(right)
	if !aOK || !bOK {
		return nil, pyError("TypeError", "unsupported operand type(s) for -: '%s' and '%s'", typeName(left), typeName(right))
	}
	if a.isFloat || b.isFloat {
		return a.asFloat() - b.asFloat(), nil
	}
	difference := a.integer - b.integer
	if (a.integer^b.integer)&(a.integer^difference) < 0 {
		return nil, errIntegerRange
	}
	return difference, nil
}

// negate is -operand.
func negate(operand any) (any, error) {
	if err := firstUndefined(operand); err != nil {
		return nil, err
	}
	if isBig(operand) {
		return nil, errBigArithmetic
	}
	n, ok := asNumber(operand)
	if !ok {
		return nil, pyError("TypeError", "bad operand type for unary -: '%s'", typeName(operand))
	}
	if n.isFloat {
		return -n.float, nil
	}
	return subtract(0, n.integer)
}

// equal is left == right.
func equal(left, right any) bool {
	if integer, ok := left.(*big.Int); ok {
		return equalsBig(integer, right)
	}
	if integer, ok := right.(*big.Int); ok {
		return equalsBig(integer, left)
	}
	if a, ok := asNumber(left); ok {
		b, ok := asNumber(right)
		if !ok {
			return false
		}
		comparison, ok := compareNumbers(a, b)
		return ok && comparison == 0
	}
	if a, ok := asText(left); ok {
		b, ok := asText(right)
		return ok && a == b
	}
	switch left := left.(type) {
	case nil:
		return right == nil
	case undefined:
		_, ok := right.(undefined)
		return ok
	case []any:
		right, ok := right.([]any)
		return ok && slices.EqualFunc(left, right, equal)
	case tuple:
		right, ok := right.(tuple)
		return ok && slices.EqualFunc(left, right, equal)
	case *dict:
		right, ok := right.(*dict)
		if !ok || len(left.keys) != len(right.keys) {
			return false
		}
		for key, value := range left.values {
			other, found := right.values[key]
			if !found || !equal(value, other) {
				return false
			}
		}
		return true
	}
	// What is left compares by identity: generators, namespaces, loops and
	// callables, all of them pointers.
	return left == right
}

// order is left op right for <, <=, > and >=.
func order(op string, left, right any) (bool, error) {
	if err := firstUndefined(left, right); err != nil {
		return false, err
	}
	if isBig(left) || isBig(right) {
		return false, errors.New("ordering an integer beyond 64 bits is not supported")
	}
	a, aNumber := asNumber(left)
	b, bNumber := asNumber(right)
	aText, aIsText := asText(left)
	bText, bIsText := asText(right)
	switch {
	case aNumber && bNumber:
		comparison, ok := compareNumbers(a, b)
		return ok && ordered(op, comparison, 0), nil
	case aIsText && bIsText:
		return ordered(op, aText, bText), nil
	}
	return false, pyError("TypeError", "'%s' not supported between instances of '%s' and '%s'", op, typeName(left), typeName(right))
}

// compareNumbers orders two numbers exactly, as Python does: an int is not
// rounded to a float to meet one. ok is false when either is NaN, which is
// neither equal to nor ordered against anything.
func compareNumbers(a, b number) (comparison int, ok bool) {
	switch {
	case !a.isFloat && !b.isFloat:
		return cmp.Compare(a.integer, b.integer), true
	case a.isFloat && b.isFloat:
		return cmp.Compare(a.float, b.float), !math.IsNaN(a.float) && !math.IsNaN(b.float)
	case a.isFloat:
		comparison, ok = compareIntFloat(b.integer, a.float)
		return -comparison, ok
	}
	return compareIntFloat(a.integer, b.float)
}

func compareIntFloat(integer int, float float64) (comparison int, ok bool) {
	switch {
	case math.IsNaN(float):
		return 0, false
	case float >= 1<<63:
		return -1, true
	case float < -(1 << 63):
		return 1, true
	}
	// The float's whole part now fits an int; its fraction breaks a tie.
	whole := math.Trunc(float)
	if comparison := cmp.Compare(integer, int(whole)); comparison != 0 {
		return comparison, true
	}
	return cmp.Compare(whole, float), true
}

func ordered[T int | float64 | string](op string, a, b T) bool {
	switch op {
	case "<":
		return a < b
	case "<=":
		return a <= b
	case ">":
		return a > b
	}
	return a >= b
}

// unhashableType reports the first unhashable type in value, in Python's
// left-to-right order (recursing into tuples).
func unhashableType(value any) (string, bool) {
	switch value := value.(type) {
	case []any:
		return "list", true
	case *dict:
		return "dict", true
	case tuple:
		for _, element := range value {
			if name, ok := unhashableType(element); ok {
				return name, true
			}
		}
	}
	return "", false
}

// contains is "item in container".
func contains(container, item any) (bool, error) {
	switch container := container.(type) {
	case undefined:
		return false, nil // iterating undefined yields nothing
	case string, safeString:
		text, _ := asText(container)
		part, ok := asText(item)
		if !ok {
			return false, pyError("TypeError", "'in <string>' requires string as left operand, not %s", typeName(item))
		}
		return strings.Contains(text, part), nil
	case []any:
		return slices.ContainsFunc(container, func(element any) bool { return equal(element, item) }), nil
	case tuple:
		return slices.ContainsFunc(container, func(element any) bool { return equal(element, item) }), nil
	case *dict:
		switch item := item.(type) {
		case string:
			_, found := container.values[item]
			return found, nil
		case safeString:
			_, found := container.values[string(item)]
			return found, nil
		case tuple:
			for _, element := range item {
				if unhashable, ok := unhashableType(element); ok {
					return false, pyError("TypeError", "unhashable type: '%s'", unhashable)
				}
			}
			return false, nil // hashable, but every key is a string
		case []any, *dict:
			return false, pyError("TypeError", "unhashable type: '%s'", typeName(item))
		}
		return false, nil // hashable, but every key is a string
	}
	return false, pyError("TypeError", "argument of type '%s' is not iterable", typeName(container))
}
