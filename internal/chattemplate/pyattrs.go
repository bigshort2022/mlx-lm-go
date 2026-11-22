package chattemplate

// pythonAttributes lists, by Python type name, the attributes a template can
// read on a value of that type in Hugging Face's sandboxed environment:
// methods such as str.upper and properties such as int.real. Any other name
// is undefined in Python, and so here.
//
// This package implements the six str methods in stringMethods. Reading one
// of the others is an error, since Python would hand the template a method
// or a number where this package has nothing to give.
//
// The lists come from dir() on Python 3.13, less the names starting with an
// underscore and the list methods that modify the list, both of which the
// sandbox hides.
var pythonAttributes = map[string][]string{
	"str": {
		"capitalize", "casefold", "center", "count", "encode", "endswith", "expandtabs", "find", "format",
		"format_map", "index", "isalnum", "isalpha", "isascii", "isdecimal", "isdigit", "isidentifier",
		"islower", "isnumeric", "isprintable", "isspace", "istitle", "isupper", "join", "ljust", "lower",
		"lstrip", "maketrans", "partition", "removeprefix", "removesuffix", "replace", "rfind", "rindex",
		"rjust", "rpartition", "rsplit", "rstrip", "split", "splitlines", "startswith", "strip", "swapcase",
		"title", "translate", "upper", "zfill",
	},
	"int":   integerAttributes,
	"bool":  integerAttributes,
	"float": {"as_integer_ratio", "conjugate", "fromhex", "hex", "imag", "is_integer", "real"},
	"list":  {"copy", "count", "index"},
	"tuple": {"count", "index"},
}

var integerAttributes = []string{
	"as_integer_ratio", "bit_count", "bit_length", "conjugate", "denominator", "from_bytes", "imag",
	"is_integer", "numerator", "real", "to_bytes",
}
