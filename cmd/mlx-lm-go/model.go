package main

import (
	"fmt"

	"github.com/bigshort2022/mlx-lm-go/internal/chattemplate"
	"github.com/bigshort2022/mlx-lm-go/internal/engine"
	"github.com/bigshort2022/mlx-lm-go/internal/generate"
	"github.com/bigshort2022/mlx-lm-go/internal/hub"
	"github.com/bigshort2022/mlx-lm-go/internal/tokenizer"
)

// model is a model directory with its tokenizer and chat template loaded.
type model struct {
	dir       string
	tokenizer *tokenizer.Tokenizer
	template  *chattemplate.Template
}

// openModel finds the model called name, a directory or a Hugging Face
// repository in the local cache, and loads its tokenizer and chat template.
// A templateSource that is not empty replaces the model's chat template.
func openModel(name, templateSource string) (*model, error) {
	dir, err := hub.Resolve(name)
	if err != nil {
		return nil, err
	}
	tok, err := tokenizer.Load(dir)
	if err != nil {
		return nil, err
	}
	var tmpl *chattemplate.Template
	if templateSource == "" {
		tmpl, err = chattemplate.Load(dir)
	} else if tmpl, err = chattemplate.LoadSource(dir, templateSource); err != nil {
		err = fmt.Errorf("--chat-template: %w", err)
	}
	if err != nil {
		return nil, err
	}
	return &model{dir: dir, tokenizer: tok, template: tmpl}, nil
}

// modelEngine is the engine of a loaded model.
type modelEngine interface {
	engine.Engine
	Close()
}

// loadEngine loads the model in dir. It is a variable because only one engine
// may be open per process and loading takes seconds: the tests load one and
// share it.
var loadEngine = func(dir string, opts generate.Options) (modelEngine, error) {
	e, err := generate.Load(dir, opts)
	if err != nil {
		return nil, err
	}
	return e, nil
}

// loadEngine loads the weights of m. The caller must Close the engine. The
// tokenizer's end-of-sequence token ends a generation when config.json names
// none.
func (m *model) loadEngine(opts generate.Options) (modelEngine, error) {
	if _, id, ok := m.tokenizer.EOSToken(); ok {
		opts.EOSTokenIDs = []int32{id}
	}
	return loadEngine(m.dir, opts)
}
