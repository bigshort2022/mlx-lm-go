package qwen3_5

import (
	"fmt"
	"strings"

	"github.com/bigshort2022/mlx-lm-go/mlx"
)

// sanitize ports the two Python sanitize passes in place: models.Load will
// evaluate exactly the tensors still present in weights after New succeeds.
func sanitize(weights map[string]*mlx.Array, tied bool) error {
	var renames map[string]string // destination -> source; pinned weights already have canonical names
	rawConv := false
	for name, a := range weights {
		if strings.HasPrefix(name, "vision_tower.") || strings.HasPrefix(name, "model.visual.") {
			a.Free()
			delete(weights, name)
			continue
		}
		canonical := name
		if suffix, ok := strings.CutPrefix(name, "model.language_model."); ok {
			canonical = textPrefix + "model." + suffix
		} else if !strings.HasPrefix(name, textPrefix) {
			canonical = textPrefix + name
		}
		// Python detects raw text convolutions before dropping MTP weights.
		rawConv = rawConv || unsanitizedConv(canonical, a)
		if strings.Contains(canonical, "mtp.") || tied && canonical == textPrefix+"lm_head.weight" {
			a.Free()
			delete(weights, name)
			continue
		}
		if canonical != name {
			if weights[canonical] != nil || renames[canonical] != "" {
				return fmt.Errorf("duplicate tensor %q after normalizing weight names", canonical)
			}
			if renames == nil {
				renames = make(map[string]string)
			}
			renames[canonical] = name
		}
	}
	for dst, src := range renames {
		weights[dst] = weights[src]
		delete(weights, src)
	}
	if !rawConv {
		return nil
	}
	return mlx.Scoped(func() error {
		var one *mlx.Array
		for name, a := range weights {
			var replacement *mlx.Array
			switch {
			case unsanitizedConv(name, a) && a.NDim() == 3:
				// Raw checkpoints use [channels, 1, kernel]; MLX uses [channels, kernel, 1].
				replacement = mlx.Transpose(a, 0, 2, 1)
			case offsetRMSWeight(name) && a.NDim() == 1:
				if one == nil {
					one = mlx.Float(1)
				}
				replacement = mlx.Add(a, mlx.AsType(one, a.DType()))
			}
			if replacement != nil {
				mlx.Keep(replacement)
				weights[name] = replacement
				a.Free()
			}
		}
		return nil
	})
}

func unsanitizedConv(name string, a *mlx.Array) bool {
	return strings.HasSuffix(name, ".conv1d.weight") && a.NDim() > 0 && a.Dim(-1) != 1
}

func offsetRMSWeight(name string) bool {
	return strings.HasSuffix(name, ".input_layernorm.weight") ||
		strings.HasSuffix(name, ".post_attention_layernorm.weight") ||
		name == textPrefix+"model.norm.weight" ||
		strings.HasSuffix(name, ".q_norm.weight") ||
		strings.HasSuffix(name, ".k_norm.weight")
}
