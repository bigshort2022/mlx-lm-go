package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// TestOpenAIClientConformance drives a server with the real model through the
// official openai Python package. It needs MLX_LM_GO_PYTHON to name a Python
// interpreter that has the package installed, and is skipped without it.
func TestOpenAIClientConformance(t *testing.T) {
	python := os.Getenv("MLX_LM_GO_PYTHON")
	if python == "" {
		t.Skip("MLX_LM_GO_PYTHON is not set")
	}
	s := startServer(t)

	script := filepath.Join("..", "..", "tools", "conformance", "openai_client.py")
	out, err := exec.Command(python, script, s.url+"/v1").CombinedOutput()
	t.Logf("openai_client.py output:\n%s", out)
	if err != nil {
		t.Fatalf("openai_client.py failed: %v", err)
	}
}
