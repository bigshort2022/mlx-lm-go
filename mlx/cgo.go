// Package mlx is a thin cgo binding over mlx-c, the C API of Apple's MLX.
//
// It is the only package in this module that imports "C". Every function in
// it must be called from one OS thread; see internal/exec.
//
// The build links Homebrew's mlx-c (brew install mlx-c). Set CGO_CFLAGS and
// CGO_LDFLAGS to use a different install prefix.
package mlx

// #cgo darwin,arm64 CFLAGS: -I/opt/homebrew/include
// #cgo darwin,arm64 LDFLAGS: -L/opt/homebrew/lib -lmlxc
import "C"
