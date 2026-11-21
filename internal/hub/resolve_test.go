package hub

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fakeCache builds <root>/models--org--name/snapshots/<rev> for each revision and,
// when ref is non-empty, points refs/main at it.
func fakeCache(t *testing.T, ref string, revs ...string) string {
	t.Helper()
	root := t.TempDir()
	repo := filepath.Join(root, "models--org--name")
	for _, rev := range revs {
		if err := os.MkdirAll(filepath.Join(repo, "snapshots", rev), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	if ref != "" {
		if err := os.MkdirAll(filepath.Join(repo, "refs"), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(repo, "refs", "main"), []byte(ref+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

func TestResolveReturnsExistingDirectoryUnchanged(t *testing.T) {
	dir := t.TempDir()
	got, err := Resolve(dir)
	if err != nil {
		t.Fatalf("Resolve(%q) error: %v", dir, err)
	}
	if got != dir {
		t.Fatalf("Resolve(%q) = %q, want the directory itself", dir, got)
	}
}

func TestResolveFollowsRefsMainInHubCache(t *testing.T) {
	root := fakeCache(t, "bbb", "aaa", "bbb")
	t.Setenv("HF_HUB_CACHE", root)
	got, err := Resolve("org/name")
	if err != nil {
		t.Fatalf("Resolve error: %v", err)
	}
	want := filepath.Join(root, "models--org--name", "snapshots", "bbb")
	if got != want {
		t.Fatalf("Resolve = %q, want %q", got, want)
	}
}

func TestResolveUsesTheOnlySnapshotWhenRefsMainIsMissing(t *testing.T) {
	root := fakeCache(t, "", "only")
	t.Setenv("HF_HUB_CACHE", root)
	got, err := Resolve("org/name")
	if err != nil {
		t.Fatalf("Resolve error: %v", err)
	}
	if filepath.Base(got) != "only" {
		t.Fatalf("Resolve = %q, want the single snapshot", got)
	}
}

func TestResolveRejectsAmbiguousSnapshotsWithoutRefsMain(t *testing.T) {
	root := fakeCache(t, "", "aaa", "bbb")
	t.Setenv("HF_HUB_CACHE", root)
	if _, err := Resolve("org/name"); err == nil {
		t.Fatal("Resolve succeeded with two snapshots and no refs/main; want an error")
	}
}

func TestResolveFallsBackToHFHomeHubDirectory(t *testing.T) {
	root := fakeCache(t, "aaa", "aaa")
	home := t.TempDir()
	if err := os.Rename(root, filepath.Join(home, "hub")); err != nil {
		t.Fatal(err)
	}
	t.Setenv("HF_HUB_CACHE", "")
	t.Setenv("HF_HOME", home)
	got, err := Resolve("org/name")
	if err != nil {
		t.Fatalf("Resolve error: %v", err)
	}
	want := filepath.Join(home, "hub", "models--org--name", "snapshots", "aaa")
	if got != want {
		t.Fatalf("Resolve = %q, want %q", got, want)
	}
}

func TestResolveMissingModelNamesTheDownloadCommand(t *testing.T) {
	t.Setenv("HF_HUB_CACHE", t.TempDir())
	_, err := Resolve("org/absent")
	if err == nil {
		t.Fatal("Resolve succeeded for a model that is not cached; want an error")
	}
	if !strings.Contains(err.Error(), "hf download org/absent") {
		t.Fatalf("error %q does not tell the user to run `hf download org/absent`", err)
	}
}
