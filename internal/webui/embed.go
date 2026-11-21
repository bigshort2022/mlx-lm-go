// Package webui holds the chat UI that is compiled into the binary, so that
// the server can offer it on its own origin.
//
// The files under dist are a build of packages/local-chat from the unite4ai
// repository. They are vendored, never edited here: scripts/sync-webui.sh
// replaces them from a checksummed tarball and records which build they are
// in VERSION and SHA256.
package webui

import (
	"embed"
	"io/fs"
	"strings"
)

//go:embed dist
var files embed.FS

//go:embed VERSION
var version string

// FS returns the embedded UI: index.html, chat.js and chat.css at its root.
func FS() fs.FS {
	ui, err := fs.Sub(files, "dist")
	if err != nil {
		panic(err) // fs.Sub only fails for an invalid directory name
	}
	return ui
}

// Version returns the pinned UI version recorded by scripts/sync-webui.sh.
func Version() string {
	return strings.TrimSpace(version)
}
