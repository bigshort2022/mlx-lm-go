// Package models defines what a loaded language model is and loads one from
// a directory. The architectures live in sub-packages, which register
// themselves; import the ones to support.
package models

import (
	"fmt"
	"maps"
	"path/filepath"
	"slices"
	"strings"

	"github.com/bigshort2022/mlx-lm-go/mlx"
)

// Model is a loaded language model. All methods must be called on the inference thread.
//
// A model should also implement VocabSize() int, the number of tokens it
// knows, and Free(), which releases its weights: package generate uses the
// first to reject prompt tokens outside the vocabulary and the second when an
// engine is closed. A model without them is not checked and not freed.
type Model interface {
	// Forward returns logits of shape [1, L, vocab] for tokens of shape [1, L],
	// reading and extending state. A nil state means no cache: the tokens are
	// the whole sequence.
	//
	// Like any operation of package mlx, Forward gives the arrays it creates
	// to the innermost open scope, those of the state included: a caller that
	// closes the scope must mlx.Keep(state.Arrays()...) first.
	Forward(tokens *mlx.Array, state State) *mlx.Array
	NewState() State
}

// State is the per-generation cache of a model.
type State interface {
	Offset() int          // tokens processed so far
	Arrays() []*mlx.Array // everything that must be evaluated to materialise the state
	Free()
}

// Builder builds the model of one architecture from its configuration and
// its tensors by name. It takes ownership of the tensors.
type Builder func(cfg Config, weights map[string]*mlx.Array) (Model, error)

// builders holds the registered architectures by model_type.
var builders = map[string]Builder{}

// Register makes an architecture available to Load under the given
// model_type. Architecture packages call it from init.
func Register(modelType string, build Builder) {
	builders[modelType] = build
}

// Load reads config.json and every *.safetensors file in dir and builds the
// model registered for its model_type. The weights are in memory when it
// returns.
func Load(dir string) (Model, Config, error) {
	cfg, err := LoadConfig(dir)
	if err != nil {
		return nil, Config{}, err
	}
	build, ok := builders[cfg.ModelType]
	if !ok {
		supported := make([]string, 0, len(builders))
		for name := range builders {
			supported = append(supported, name)
		}
		slices.Sort(supported)
		return nil, Config{}, fmt.Errorf("model type %q is not supported; supported: %s",
			cfg.ModelType, strings.Join(supported, ", "))
	}
	weights, err := loadWeights(dir)
	if err != nil {
		return nil, Config{}, err
	}
	model, err := build(cfg, weights)
	if err != nil {
		return nil, Config{}, err
	}
	// The tensors are read from the files only now, on the thread that
	// opened them, as MLX requires.
	mlx.Eval(slices.Collect(maps.Values(weights))...)
	return model, cfg, nil
}

// loadWeights lazily loads the tensors of every *.safetensors file in dir.
func loadWeights(dir string) (map[string]*mlx.Array, error) {
	files, err := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	if err != nil {
		return nil, err
	}
	if len(files) == 0 {
		return nil, fmt.Errorf("no safetensors found in %s", dir)
	}
	weights := map[string]*mlx.Array{}
	for _, file := range files {
		tensors, _, err := mlx.LoadSafetensors(file)
		if err != nil {
			for _, a := range weights {
				a.Free()
			}
			return nil, fmt.Errorf("%s: %w", file, err)
		}
		for name, a := range tensors {
			if old, ok := weights[name]; ok {
				old.Free()
			}
			weights[name] = a
		}
	}
	return weights, nil
}
