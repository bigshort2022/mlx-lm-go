package chattemplate

// isPythonSpace reports whether Python's str.isspace, and so str.strip and
// the regular expression class \s, treat the character as whitespace. Go's
// unicode.IsSpace differs: it leaves out U+001C to U+001F.
func isPythonSpace(char rune) bool {
	switch {
	case char >= 0x09 && char <= 0x0d, char >= 0x1c && char <= 0x20:
		return true
	case char == 0x85, char == 0xa0, char == 0x1680:
		return true
	case char >= 0x2000 && char <= 0x200a:
		return true
	case char == 0x2028, char == 0x2029, char == 0x202f, char == 0x205f, char == 0x3000:
		return true
	}
	return false
}
