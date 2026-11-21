package main

import (
	"bytes"
	"context"
	"regexp"
	"strings"
	"testing"
)

// runCommand executes one command line and returns its exit code and output.
func runCommand(t *testing.T, args ...string) (code int, stdout, stderr string) {
	t.Helper()
	var out, errs bytes.Buffer
	code = run(context.Background(), args, &out, &errs)
	return code, out.String(), errs.String()
}

func TestUsageErrorsExitWith2AndExplain(t *testing.T) {
	tests := []struct {
		name   string
		args   []string
		reason string
	}{
		{"no command", nil, "no command given"},
		{"unknown command", []string{"serve"}, `unknown command "serve"`},
		{"unsupported server flag", []string{"server", "--model", "m", "--adapter-path", "a"}, "--adapter-path is not supported yet"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			code, stdout, stderr := runCommand(t, tt.args...)
			if code != 2 {
				t.Errorf("exit code = %d, want 2", code)
			}
			if stdout != "" {
				t.Errorf("stdout = %q, want nothing", stdout)
			}
			if !strings.Contains(stderr, tt.reason) {
				t.Errorf("stderr does not give the reason %q:\n%s", tt.reason, stderr)
			}
			if !strings.Contains(stderr, "usage: mlx-lm-go ") {
				t.Errorf("stderr has no usage hint:\n%s", stderr)
			}
		})
	}
}

func TestVersionPrintsTheVersionLine(t *testing.T) {
	code, stdout, stderr := runCommand(t, "version")
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr:\n%s", code, stderr)
	}
	line := regexp.MustCompile(`^mlx-lm-go \S+ \(MLX \d+\.\d+\.\d+, UI \S+\)\n$`)
	if !line.MatchString(stdout) {
		t.Errorf("stdout = %q, want a line matching %s", stdout, line)
	}
	if !strings.HasPrefix(stdout, "mlx-lm-go dev ") {
		t.Errorf("stdout = %q, want the version of an unstamped build to be dev", stdout)
	}
}
