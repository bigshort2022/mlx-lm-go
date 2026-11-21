package main

import (
	"context"
	"fmt"
	"os"
	"sync"
	"testing"

	"github.com/bigshort2022/mlx-lm-go/internal/engine"
	"github.com/bigshort2022/mlx-lm-go/internal/generate"
	"github.com/bigshort2022/mlx-lm-go/internal/hub"
	"github.com/bigshort2022/mlx-lm-go/internal/tokenizer"
)

// referenceModel is the model the recordings in testdata/server were made
// with.
const referenceModel = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

// needModel skips the test when the reference model is not on this machine.
func needModel(t *testing.T) {
	t.Helper()
	if _, err := hub.Resolve(referenceModel); err != nil {
		t.Skip(err)
	}
}

// referenceTokenizer loads the tokenizer of the reference model.
func referenceTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	needModel(t)
	dir, err := hub.Resolve(referenceModel)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := tokenizer.Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

// sharedModel is the one engine of the test binary together with a record of
// what the commands under test did with it. Nothing frees the weights of a
// loaded model, so the commands must not each load their own.
type sharedModel struct {
	once   sync.Once
	dir    string
	engine *generate.Engine
	err    error

	mu      sync.Mutex
	opts    generate.Options // of the latest load
	request engine.Request   // the latest one generated
	result  engine.Result    // of the latest generation
	closes  int              // times a command closed the engine it was given
}

var shared sharedModel

func TestMain(m *testing.M) {
	loadEngine = shared.load
	code := m.Run()
	if shared.engine != nil {
		shared.engine.Close()
	}
	os.Exit(code)
}

// load stands in for loadEngine: the first call loads the model and every
// call returns that engine, which stays open until TestMain ends.
func (s *sharedModel) load(dir string, opts generate.Options) (modelEngine, error) {
	s.once.Do(func() {
		s.dir = dir
		s.engine, s.err = generate.Load(dir, opts)
	})
	if s.err != nil {
		return nil, s.err
	}
	if dir != s.dir {
		return nil, fmt.Errorf("the tests share the model in %s and cannot also load %s", s.dir, dir)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.opts = opts
	return sharedEngine{s}, nil
}

// lastOptions returns the options of the latest load.
func (s *sharedModel) lastOptions() generate.Options {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.opts
}

// lastRequest returns the latest request the engine was given.
func (s *sharedModel) lastRequest() engine.Request {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.request
}

// lastResult returns the result of the latest generation that has ended.
func (s *sharedModel) lastResult() engine.Result {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.result
}

// closeCount returns how many times a command has closed its engine.
func (s *sharedModel) closeCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.closes
}

// sharedEngine is the shared engine as one command sees it.
type sharedEngine struct{ model *sharedModel }

func (e sharedEngine) Generate(ctx context.Context, req engine.Request, emit func(token int32)) (engine.Result, error) {
	s := e.model
	s.mu.Lock()
	s.request = req
	s.mu.Unlock()

	res, err := s.engine.Generate(ctx, req, emit)

	s.mu.Lock()
	s.result = res
	s.mu.Unlock()
	return res, err
}

func (e sharedEngine) Close() {
	e.model.mu.Lock()
	defer e.model.mu.Unlock()
	e.model.closes++
}
