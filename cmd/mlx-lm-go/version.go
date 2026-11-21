package main

import (
	"fmt"
	"io"

	"github.com/bigshort2022/mlx-lm-go/internal/webui"
	"github.com/bigshort2022/mlx-lm-go/mlx"
)

// version names the build. Releases set it with
// -ldflags "-X main.version=<version>".
var version = "dev"

// runVersion prints the versions of the binary, the MLX library it is linked
// to and the embedded UI.
func runVersion(stdout, stderr io.Writer) int {
	mlxVersion, err := mlx.Version()
	if err != nil {
		return failure(stderr, err)
	}
	fmt.Fprintf(stdout, "mlx-lm-go %s (MLX %s, UI %s)\n", version, mlxVersion, webui.Version())
	return exitOK
}
