package webui

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/pem"
	"io/fs"
	"maps"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// entry is one member of a test tarball. A non-empty link makes it a symlink.
type entry struct {
	name, body, link string
}

// goodUI is a tarball laid out like a release of the UI.
var goodUI = []entry{
	{name: "index.html", body: "<!doctype html><title>chat</title>"},
	{name: "chat.js", body: "console.log('chat')"},
	{name: "chat.css", body: "body { margin: 0 }"},
}

// writeTarball writes a gzipped tarball of the entries into dir and returns
// its path and SHA-256.
func writeTarball(t *testing.T, dir string, entries []entry) (path, sum string) {
	t.Helper()
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gz)
	for _, e := range entries {
		hdr := &tar.Header{Name: e.name, Mode: 0o644, Size: int64(len(e.body)), Typeflag: tar.TypeReg}
		if e.link != "" {
			hdr.Typeflag, hdr.Linkname, hdr.Size = tar.TypeSymlink, e.link, 0
		}
		if err := tw.WriteHeader(hdr); err != nil {
			t.Fatal(err)
		}
		if e.link == "" {
			if _, err := tw.Write([]byte(e.body)); err != nil {
				t.Fatal(err)
			}
		}
	}
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := gz.Close(); err != nil {
		t.Fatal(err)
	}
	path = filepath.Join(dir, "ui.tar.gz")
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(buf.Bytes())
	return path, hex.EncodeToString(digest[:])
}

// runSync runs scripts/sync-webui.sh with target standing in for
// internal/webui and returns what it printed and how it ended.
func runSync(t *testing.T, target string, args ...string) (string, error) {
	t.Helper()
	script, err := filepath.Abs(filepath.Join("..", "..", "scripts", "sync-webui.sh"))
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command("sh", append([]string{script}, args...)...)
	cmd.Env = append(os.Environ(), "WEBUI_DIR="+target)
	cmd.Dir = t.TempDir() // the script must not depend on where it is run from
	out, err := cmd.CombinedOutput()
	return string(out), err
}

// tree returns the content of every file under root by relative path, with
// directories and symlinks as entries of their own.
func tree(t *testing.T, root string) map[string]string {
	t.Helper()
	files := map[string]string{}
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(root, path)
		switch {
		case d.IsDir():
			files[rel+"/"] = ""
		case d.Type()&fs.ModeSymlink != 0:
			files[rel] = "symlink"
		default:
			body, err := os.ReadFile(path)
			files[rel] = string(body)
			return err
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return files
}

// existingUI creates a target directory that already holds an older UI inside
// a parent directory of its own, and returns both.
func existingUI(t *testing.T) (parent, target string) {
	t.Helper()
	parent = t.TempDir()
	target = filepath.Join(parent, "webui")
	for name, body := range map[string]string{
		"dist/index.html": "old page",
		"dist/chat.js":    "old script",
		"dist/chat.css":   "old style",
		"dist/old.txt":    "left over from the old build",
		"VERSION":         "0.9.0\n",
		"SHA256":          strings.Repeat("0", 64) + "\n",
		"embed.go":        "package webui",
	} {
		path := filepath.Join(target, name)
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return parent, target
}

func TestSyncReplacesDistAndRecordsVersionAndChecksum(t *testing.T) {
	_, target := existingUI(t)
	tarball, sum := writeTarball(t, t.TempDir(), goodUI)

	if out, err := runSync(t, target, tarball, sum, "1.2.3"); err != nil {
		t.Fatalf("sync failed: %v\n%s", err, out)
	}
	want := map[string]string{
		"./":              "",
		"dist/":           "",
		"dist/index.html": goodUI[0].body,
		"dist/chat.js":    goodUI[1].body,
		"dist/chat.css":   goodUI[2].body,
		"VERSION":         "1.2.3\n",
		"SHA256":          sum + "\n",
		"embed.go":        "package webui",
	}
	if got := tree(t, target); !maps.Equal(got, want) {
		t.Fatalf("target holds\n%q\nwant\n%q", got, want)
	}
}

// assertRefused runs the sync, which must fail, mention why and leave the
// parent of the target directory exactly as it was.
func assertRefused(t *testing.T, entries []entry, sumOverride, reason string) {
	t.Helper()
	parent, target := existingUI(t)
	before := tree(t, parent)
	tarball, sum := writeTarball(t, t.TempDir(), entries)
	if sumOverride != "" {
		sum = sumOverride
	}

	out, err := runSync(t, target, tarball, sum, "1.2.3")
	if err == nil {
		t.Fatalf("sync succeeded, want a failure:\n%s", out)
	}
	if !strings.Contains(out, reason) {
		t.Errorf("output does not mention %q:\n%s", reason, out)
	}
	if after := tree(t, parent); !maps.Equal(after, before) {
		t.Errorf("a refused sync changed files:\n%q\nwas\n%q", after, before)
	}
}

func TestSyncRefusesAWrongChecksum(t *testing.T) {
	assertRefused(t, goodUI, strings.Repeat("ab", 32), "checksum")
}

func TestSyncRefusesATarballWithoutARequiredFile(t *testing.T) {
	for i, missing := range goodUI {
		t.Run(missing.name, func(t *testing.T) {
			var entries []entry
			entries = append(entries, goodUI[:i]...)
			entries = append(entries, goodUI[i+1:]...)
			assertRefused(t, entries, "", missing.name)
		})
	}
}

func TestSyncRefusesPathsOutsideTheTarballRoot(t *testing.T) {
	for _, name := range []string{"../evil", "assets/../../evil", "/tmp/mlx-lm-go-sync-test-evil"} {
		t.Run(name, func(t *testing.T) {
			if filepath.IsAbs(name) {
				t.Cleanup(func() { os.Remove(name) })
			}
			assertRefused(t, append([]entry{{name: name, body: "escaped"}}, goodUI...), "", "outside")
			if _, err := os.Lstat(name); filepath.IsAbs(name) && err == nil {
				t.Errorf("%s was written", name)
			}
		})
	}
}

func TestSyncRefusesLinks(t *testing.T) {
	// A link could point outside the root even though its own name does not.
	entries := append([]entry{{name: "assets", link: "/tmp"}}, goodUI...)
	assertRefused(t, entries, "", "regular files")
}

func TestSyncDownloadsAnHTTPSURL(t *testing.T) {
	served := t.TempDir()
	_, sum := writeTarball(t, served, goodUI)
	ts := httptest.NewTLSServer(http.FileServer(http.Dir(served)))
	defer ts.Close()
	// Make curl trust the test server's certificate.
	ca := filepath.Join(t.TempDir(), "ca.pem")
	block := &pem.Block{Type: "CERTIFICATE", Bytes: ts.Certificate().Raw}
	if err := os.WriteFile(ca, pem.EncodeToMemory(block), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("CURL_CA_BUNDLE", ca)

	_, target := existingUI(t)
	if out, err := runSync(t, target, ts.URL+"/ui.tar.gz", sum, "2.0.0"); err != nil {
		t.Fatalf("sync failed: %v\n%s", err, out)
	}
	got := tree(t, target)
	if got["dist/index.html"] != goodUI[0].body || got["VERSION"] != "2.0.0\n" || got["SHA256"] != sum+"\n" {
		t.Fatalf("target holds %q, want the downloaded UI at version 2.0.0", got)
	}
}

func TestSyncRefusesToDownloadWithoutTLS(t *testing.T) {
	served := t.TempDir()
	_, sum := writeTarball(t, served, goodUI)
	ts := httptest.NewServer(http.FileServer(http.Dir(served)))
	defer ts.Close()

	parent, target := existingUI(t)
	before := tree(t, parent)
	out, err := runSync(t, target, ts.URL+"/ui.tar.gz", sum, "2.0.0")
	if err == nil || !strings.Contains(out, "https") {
		t.Fatalf("sync of an http URL: err = %v, output:\n%s\nwant a refusal that asks for https", err, out)
	}
	if after := tree(t, parent); !maps.Equal(after, before) {
		t.Errorf("a refused sync changed files:\n%q\nwas\n%q", after, before)
	}
}

func TestSyncRefusesAMissingTarball(t *testing.T) {
	parent, target := existingUI(t)
	before := tree(t, parent)
	missing := filepath.Join(t.TempDir(), "nope.tar.gz")
	out, err := runSync(t, target, missing, strings.Repeat("ab", 32), "1.2.3")
	if err == nil || !strings.Contains(out, "no such tarball") {
		t.Fatalf("err = %v, output:\n%s\nwant a refusal naming the missing tarball", err, out)
	}
	if after := tree(t, parent); !maps.Equal(after, before) {
		t.Errorf("a refused sync changed files")
	}
}

func TestSyncNeedsThreeArguments(t *testing.T) {
	_, target := existingUI(t)
	out, err := runSync(t, target, "only-one")
	if err == nil || !strings.Contains(out, "usage:") {
		t.Fatalf("err = %v, output:\n%s\nwant the usage message", err, out)
	}
}
