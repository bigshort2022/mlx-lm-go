package mlx

import (
	"regexp"
	"testing"
)

func TestVersionReportsLinkedMLX(t *testing.T) {
	got, err := Version()
	if err != nil {
		t.Fatalf("Version() error: %v", err)
	}
	if !regexp.MustCompile(`^\d+\.\d+\.\d+`).MatchString(got) {
		t.Fatalf("Version() = %q, want a semantic version from the linked MLX library", got)
	}
}
