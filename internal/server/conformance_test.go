package server

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TestOpenAIClientConformance drives the server with the official openai
// Python package. It needs MLX_LM_GO_PYTHON to name a Python interpreter that
// has the package installed, and is skipped without it.
func TestOpenAIClientConformance(t *testing.T) {
	python := os.Getenv("MLX_LM_GO_PYTHON")
	if python == "" {
		t.Skip("MLX_LM_GO_PYTHON is not set")
	}
	_, ts := newLiveHarness(t, scripted(countPieces...))

	script := filepath.Join("..", "..", "tools", "conformance", "openai_client.py")
	out, err := exec.Command(python, script, ts.URL+"/v1").CombinedOutput()
	t.Logf("openai_client.py output:\n%s", out)
	if err != nil {
		t.Fatalf("openai_client.py failed: %v", err)
	}
}

// TestOpenAIClientConformanceReportsFailures checks the script itself: against
// a server whose model says nothing it must fail and say why.
func TestOpenAIClientConformanceReportsFailures(t *testing.T) {
	python := os.Getenv("MLX_LM_GO_PYTHON")
	if python == "" {
		t.Skip("MLX_LM_GO_PYTHON is not set")
	}
	_, ts := newLiveHarness(t, scripted())

	script := filepath.Join("..", "..", "tools", "conformance", "openai_client.py")
	out, err := exec.Command(python, script, ts.URL+"/v1").CombinedOutput()
	var exit *exec.ExitError
	if !errors.As(err, &exit) || exit.ExitCode() == 0 {
		t.Fatalf("script error = %v, want a non-zero exit:\n%s", err, out)
	}
	if !strings.Contains(string(out), "FAIL: chat content is empty") {
		t.Fatalf("script output does not name the failed check:\n%s", out)
	}
}
