// Package deps pins third-party modules in go.mod while the packages that use
// them are being written. Delete it once every listed module has a real importer.
package deps

import (
	_ "github.com/dlclark/regexp2"
	_ "github.com/nikolalohinski/gonja/v2"
	_ "golang.org/x/text/unicode/norm"
)
