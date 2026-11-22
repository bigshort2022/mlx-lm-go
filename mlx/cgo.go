// Package mlx is a thin cgo binding over mlx-c, the C API of Apple's MLX.
//
// It is the only package in this module that imports "C".
//
// Only one OS thread per process may use the package at a time: the stack of
// scopes and the pending MLX error are process-wide and not synchronised. MLX
// itself keeps more state per thread (streams, the random key), so a program
// makes every call from one thread, the one internal/exec provides. The Array
// documentation says which arrays may move to another thread.
//
// The build links Homebrew's mlx-c (brew install mlx-c). Set CGO_CFLAGS and
// CGO_LDFLAGS to use a different install prefix.
package mlx

// #cgo darwin,arm64 CFLAGS: -I/opt/homebrew/include
// #cgo darwin,arm64 LDFLAGS: -L/opt/homebrew/lib -lmlxc
import "C"
