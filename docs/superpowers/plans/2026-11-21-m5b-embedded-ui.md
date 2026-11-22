# M5b: the chat UI embedded in the binary

> **For agentic workers:** implement task by task with red/green TDD. For every test: write it, run it, see it fail for the stated reason, then write the smallest code that passes. Steps use checkbox (`- [ ]`) syntax.

**Goal:** `http://127.0.0.1:<port>/` serves the same chat UI that unite4ai.com hosts, from files compiled into the binary, so the UI works in browsers that block a public page from calling loopback.

**Architecture:** the UI is built in the unite4ai repository (`packages/local-chat`). This repository vendors a pinned, checksummed copy under `internal/webui/dist` and embeds it with `go:embed`. `internal/server` serves it when given a file system.

**Tech stack:** Go 1.27 standard library (`embed`, `io/fs`, `net/http`), POSIX shell, `shasum`, `tar`.

---

## Starting point

- The UI build is produced by `bun packages/local-chat/build.ts` in `/Users/b33j_02/sandbox/unite4ai.github.io` and lands in `packages/local-chat/dist/{index.html,chat.js,chat.css}`. Do not edit that repository; run its build and read its output only.
- `internal/server` already exists with `New(cfg Config, deps Deps) http.Handler`, Host and Origin checks, and JSON routes. Read `internal/server/server.go` and `security.go` first.

## Files you own

Create: `scripts/sync-webui.sh`, `internal/webui/embed.go`, `internal/webui/embed_test.go`, `internal/webui/sync_test.go`, `internal/webui/VERSION`, `internal/webui/SHA256`, `internal/webui/dist/*`, `internal/server/webui.go`, `internal/server/webui_test.go`.
Modify: `internal/server/server.go` (routing only) and the `Deps` struct (one new field).
Do not touch anything else.

## Contract

```go
package webui

// FS returns the embedded UI: index.html, chat.js and chat.css at its root.
func FS() fs.FS
// Version returns the pinned UI version recorded by scripts/sync-webui.sh.
func Version() string
```

```go
package server

type Deps struct {
	// ... existing fields ...
	WebUI fs.FS // nil: no UI is served
}
```

`scripts/sync-webui.sh <tarball path or https URL> <sha256> <version>`:

- downloads the tarball when given a URL, otherwise reads the file;
- exits non-zero, changing nothing, when the SHA-256 does not match;
- replaces `internal/webui/dist` with the tarball's contents and writes `VERSION` and `SHA256`;
- exits non-zero, changing nothing, when the tarball lacks `index.html`, `chat.js` or `chat.css`, or contains any path outside its root (`..` or absolute).

## Tasks

Run tests with `go test ./internal/webui/ -run <Name> -v` and `go test -race ./internal/server/ -run <Name> -v`. Never run `go build ./...`, `go test ./...`, `go mod tidy` or `go get`.

### Task 1: the sync script

Tests run the script through `os/exec` with a temp directory standing in for `internal/webui` (give the script an optional `WEBUI_DIR` environment variable for that).

- [ ] A good tarball with the right checksum replaces `dist` and writes `VERSION` and `SHA256`.
- [ ] A wrong checksum exits non-zero and leaves an existing `dist` byte-identical.
- [ ] A tarball missing `chat.js` exits non-zero and changes nothing.
- [ ] A tarball containing `../evil` exits non-zero and writes nothing outside the target directory.

### Task 2: vendor the build

- [ ] In the unite4ai checkout run `bun packages/local-chat/build.ts`, pack `packages/local-chat/dist` into a tarball in a temp directory, and run the sync script with version `dev-<short commit of the unite4ai checkout>`.
- [ ] `TestEmbeddedUIHasItsThreeFiles`: `FS()` has non-empty `index.html`, `chat.js`, `chat.css`.
- [ ] `TestEmbeddedUIMakesNoNetworkReferences`: none of the three files contains `https://` or `http://` other than inside the string `http://127.0.0.1` used as a placeholder or example.
- [ ] `TestVersionMatchesVersionFile`.

### Task 3: serve it

- [ ] `GET /` with `Deps.WebUI` set returns `index.html` with `Content-Type: text/html; charset=utf-8`.
- [ ] `/chat.js` and `/chat.css` are served with JavaScript and CSS content types.
- [ ] Every UI response carries `Content-Security-Policy: default-src 'self'; base-uri 'none'; form-action 'none'` and `X-Content-Type-Options: nosniff`.
- [ ] With `Deps.WebUI == nil`, `GET /` is 404 as before.
- [ ] `GET /nope` is still 404, and `/v1/models` and `/health` behave as before.
- [ ] The Host check applies to UI routes: a wrong `Host` header returns 403.
- [ ] Only `GET` and `HEAD` reach the UI; `POST /` is 404 or 405, never a file.

## Done when

`go test ./internal/webui/ && go test -race ./internal/server/ && go vet ./internal/webui/ ./internal/server/` passes with clean output, `gofmt -l internal/webui internal/server` prints nothing, and `sh -n scripts/sync-webui.sh` succeeds.

## Report

For each task: the test names, the failure you saw before implementing (one line each), and the final pass. Give the vendored version and checksum. Do not commit.
