// Command mlx-lm-go serves an MLX language model over an OpenAI-compatible
// HTTP API, as Python's mlx_lm.server does, and answers single prompts.
//
//	mlx-lm-go server --model <dir or repo id> [flags of mlx_lm.server]
//	mlx-lm-go generate --model <dir or repo id> --prompt <text> [--system <text>]
//	        [--max-tokens N] [--temp T] [--chat-template-args <json>]
//	mlx-lm-go version
package main

import (
	"context"
	"os"
	"os/signal"
	"syscall"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	// A second signal ends the process at once.
	go func() {
		<-ctx.Done()
		stop()
	}()
	os.Exit(run(ctx, os.Args[1:], os.Stdout, os.Stderr))
}
