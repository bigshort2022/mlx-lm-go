package main

import (
	"bytes"
	"errors"
	"os/exec"
	"path/filepath"
	"regexp"
	"testing"
)

// TestBinary builds the command the way a release does and runs it as a
// process. It does not load a model.
func TestBinary(t *testing.T) {
	binary := filepath.Join(t.TempDir(), "mlx-lm-go")
	build := exec.Command("go", "build", "-ldflags", "-X main.version=1.2.3", "-o", binary, ".")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("go build: %v\n%s", err, out)
	}

	t.Run("version prints the version set at build time", func(t *testing.T) {
		out, err := exec.Command(binary, "version").CombinedOutput()
		if err != nil {
			t.Fatalf("%v\n%s", err, out)
		}
		line := regexp.MustCompile(`^mlx-lm-go 1\.2\.3 \(MLX \d+\.\d+\.\d+, UI \S+\)\n$`)
		if !line.Match(out) {
			t.Errorf("output = %q, want a line matching %s", out, line)
		}
	})

	t.Run("a usage error exits with 2 and writes to stderr only", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		cmd := exec.Command(binary, "serve")
		cmd.Stdout, cmd.Stderr = &stdout, &stderr
		err := cmd.Run()
		var exit *exec.ExitError
		if !errors.As(err, &exit) || exit.ExitCode() != 2 {
			t.Errorf("error = %v, want exit status 2", err)
		}
		if stdout.Len() != 0 || !bytes.Contains(stderr.Bytes(), []byte(`unknown command "serve"`)) {
			t.Errorf("stdout = %q, stderr = %q, want the reason on stderr only", &stdout, &stderr)
		}
	})
}
