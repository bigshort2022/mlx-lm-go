package main

import (
	"context"
	"fmt"
	"io"
)

// Exit codes of the process.
const (
	exitOK      = 0
	exitFailure = 1 // the command could not do its work
	exitUsage   = 2 // the command line is wrong
)

const usageHint = "usage: mlx-lm-go server|generate|version [flags]"

// run executes one command line and returns the process exit code.
// ctx cancellation shuts a running server down.
func run(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		return usageError(stderr, "no command given")
	}
	switch command, args := args[0], args[1:]; command {
	case "server":
		return runServer(ctx, args, stderr)
	case "generate":
		return runGenerate(ctx, args, stdout, stderr)
	case "version":
		return runVersion(stdout, stderr)
	default:
		return usageError(stderr, fmt.Sprintf("unknown command %q", command))
	}
}

// usageError reports a wrong command line and returns its exit code.
func usageError(stderr io.Writer, reason string) int {
	fmt.Fprintf(stderr, "mlx-lm-go: %s\n%s\n", reason, usageHint)
	return exitUsage
}

// failure reports why a command could not do its work and returns its exit
// code.
func failure(stderr io.Writer, err error) int {
	fmt.Fprintf(stderr, "mlx-lm-go: %v\n", err)
	return exitFailure
}
