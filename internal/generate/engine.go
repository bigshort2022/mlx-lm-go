// Package generate runs a loaded model: it owns the inference thread, feeds
// the prompt in steps and samples tokens until the model stops. It is a port
// of generate_step in mlx_lm/generate.py behind the engine.Engine interface.
package generate

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"
	"sync/atomic"

	"github.com/bigshort2022/mlx-lm-go/internal/engine"
	"github.com/bigshort2022/mlx-lm-go/internal/exec"
	"github.com/bigshort2022/mlx-lm-go/internal/models"
	_ "github.com/bigshort2022/mlx-lm-go/internal/models/qwen2"
	_ "github.com/bigshort2022/mlx-lm-go/internal/models/qwen3_5"
	"github.com/bigshort2022/mlx-lm-go/internal/sample"
	"github.com/bigshort2022/mlx-lm-go/mlx"
)

// defaultPrefillStepSize is prefill_step_size of the reference.
const defaultPrefillStepSize = 2048

// clearCacheEvery is the number of tokens between two clearings of MLX's
// buffer cache, as in the reference.
const clearCacheEvery = 256

// Options configures an Engine.
type Options struct {
	PrefillStepSize int     // 0 means 2048
	EOSTokenIDs     []int32 // used only when config.json has none
}

// Engine implements engine.Engine. It owns one inference thread.
type Engine struct {
	thread          *exec.Thread
	model           models.Model
	vocab           int // token ids run from 0 up to but not including it; 0 if the model does not tell
	eos             []int32
	prefillStepSize int

	loaded    bool        // made by Load: it holds engineOpen until Close
	closing   atomic.Bool // set by Close: no generation starts any more
	closeOnce sync.Once
}

// engineOpen is true while an engine of this process is open. There can be
// only one: the scopes of package mlx are process-wide, so two inference
// threads would free each other's arrays.
var engineOpen atomic.Bool

// vocabulary is implemented by the models that know how many tokens they
// have.
type vocabulary interface{ VocabSize() int }

var _ engine.Engine = (*Engine)(nil)

// Load loads the model in dir onto a new inference thread. Only one engine
// can be open in a process: Load fails until the previous one is closed.
func Load(dir string, opts Options) (*Engine, error) {
	if !engineOpen.CompareAndSwap(false, true) {
		return nil, errors.New("generate: an engine is already open; close it first")
	}
	e := &Engine{thread: exec.Start(), prefillStepSize: opts.PrefillStepSize, loaded: true}
	if e.prefillStepSize <= 0 {
		e.prefillStepSize = defaultPrefillStepSize
	}
	err := e.thread.Do(context.Background(), func() error {
		model, cfg, err := models.Load(dir)
		if err != nil {
			return err
		}
		e.model = model
		if v, ok := model.(vocabulary); ok {
			e.vocab = v.VocabSize()
		}
		e.eos = endOfSequence(cfg, opts)
		return nil
	})
	if err != nil {
		e.thread.Close()
		engineOpen.Store(false)
		return nil, err
	}
	return e, nil
}

// endOfSequence returns the tokens that end a generation: those of
// config.json, or those of the options when it names none.
func endOfSequence(cfg models.Config, opts Options) []int32 {
	if len(cfg.EOSTokenIDs) > 0 {
		return cfg.EOSTokenIDs
	}
	return opts.EOSTokenIDs
}

// releasable is implemented by the models that can free their weights.
type releasable interface{ Free() }

// Close frees the model and stops the inference thread, after the generation
// in progress. Generations that wait behind it and later calls of Generate
// fail with exec.ErrClosed. Once Close has returned, Load can open another
// engine.
func (e *Engine) Close() {
	e.closeOnce.Do(func() {
		e.closing.Store(true)
		// Arrays have no finalizers: the weights stay until they are freed,
		// and that must happen on the thread that owns them. An error means
		// the thread is gone, and with it the chance to free.
		_ = e.thread.Do(context.Background(), func() error {
			if model, ok := e.model.(releasable); ok {
				model.Free()
			}
			mlx.ClearCache()
			return nil
		})
		e.thread.Close()
		if e.loaded {
			engineOpen.Store(false)
		}
	})
}

// Generate implements engine.Engine. It may be called from any goroutine;
// calls run one at a time, in arrival order.
//
// It samples at most req.MaxTokens tokens. It finishes with "stop" when it
// samples an end-of-sequence token, which it counts in CompletionTokens but
// does not emit, and with "length" when the budget runs out first.
//
// When ctx ends it returns ctx.Err() together with the counts reached so far.
// A single MLX evaluation cannot be interrupted, so that takes up to one
// token while tokens are sampled and up to one prefill step while the prompt
// is fed: about a second for the default step of 2048 tokens on the
// reference model. A smaller Options.PrefillStepSize shortens it.
//
// It fails, before any token, for an empty prompt, for a prompt token
// outside the vocabulary and for sampling parameters the reference rejects.
//
// emit runs on the inference thread: it must not call Generate or Close.
func (e *Engine) Generate(ctx context.Context, req engine.Request, emit func(token int32)) (engine.Result, error) {
	res := engine.Result{PromptTokens: len(req.Prompt)}
	err := e.thread.Do(ctx, func() error {
		if e.closing.Load() {
			return exec.ErrClosed // queued before Close: the model is being freed
		}
		return e.generate(ctx, req, emit, &res)
	})
	return res, err
}

// generate runs one request on the inference thread. It keeps res up to date
// so that the counts survive an error.
func (e *Engine) generate(ctx context.Context, req engine.Request, emit func(token int32), res *engine.Result) error {
	if len(req.Prompt) == 0 {
		return errors.New("generate: the prompt is empty")
	}
	if err := e.checkPrompt(req.Prompt); err != nil {
		return err
	}
	if req.MaxTokens <= 0 {
		res.FinishReason = "length"
		return nil
	}
	sampler, err := sample.New(req.Sampling)
	if err != nil {
		return err
	}
	run := &run{model: e.model, state: e.model.NewState(), sampler: sampler}
	defer run.free()

	// Feed all but the last prompt token in steps, to bound the memory of
	// one forward pass.
	prompt := req.Prompt
	for len(prompt) > 1 {
		if err := ctx.Err(); err != nil {
			return err
		}
		n := min(e.prefillStepSize, len(prompt)-1)
		if err := run.prefill(prompt[:n]); err != nil {
			return err
		}
		prompt = prompt[n:]
		mlx.ClearCache()
	}

	// Like the reference, compute one token ahead: the next step is in flight
	// on the GPU while the current token is read and handed to emit.
	if run.y, err = run.first(prompt); err != nil {
		return err
	}
	for n := 0; ; n++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		if n+1 < req.MaxTokens {
			if run.next, err = run.step(run.y); err != nil {
				return err
			}
		}
		token := run.y.Int32s()[0] // waits for the token
		res.CompletionTokens++
		if slices.Contains(e.eos, token) {
			res.FinishReason = "stop"
			return nil
		}
		emit(token)
		if run.next == nil {
			res.FinishReason = "length"
			return nil
		}
		if n%clearCacheEvery == 0 {
			mlx.ClearCache()
		}
		run.y.Free()
		run.y, run.next = run.next, nil
	}
}

// checkPrompt reports the first prompt token that is not in the vocabulary.
// MLX checks no index: such a token would be embedded from memory outside
// the table and, with a penalty, used to write outside the logits.
func (e *Engine) checkPrompt(prompt []int32) error {
	if e.vocab == 0 {
		return nil
	}
	for position, id := range prompt {
		if id < 0 || int(id) >= e.vocab {
			return fmt.Errorf("generate: prompt token %d at position %d is outside the vocabulary of %d tokens", id, position, e.vocab)
		}
	}
	return nil
}

// run is the state of one generation. It owns its arrays.
type run struct {
	model   models.Model
	state   models.State
	sampler *sample.Sampler
	tokens  *mlx.Array // the tokens so far, kept only when the sampler uses them
	y       *mlx.Array // the token being computed or read: int32, shape [1]
	next    *mlx.Array // the token after y, computed ahead; nil at the last one
}

// free releases what the run holds, however the generation ended.
func (r *run) free() {
	r.state.Free()
	for _, a := range []*mlx.Array{r.tokens, r.y, r.next} {
		if a != nil {
			a.Free()
		}
	}
}

// The forward passes below are built inside a scope and evaluated only after
// it has closed. Closing it drops the handles of the intermediate arrays, as
// returning from the model does in Python, so that MLX can release each
// activation as soon as the evaluation has passed it. With the handles alive
// every activation of every layer would stay until the end of the pass.

// prefill feeds prompt tokens to the model and materialises its state.
func (r *run) prefill(prompt []int32) error {
	err := mlx.Scoped(func() error {
		processed := mlx.FromInt32(prompt, len(prompt))
		r.model.Forward(mlx.ExpandDims(processed, 0), r.state)
		mlx.Keep(r.state.Arrays()...)
		r.remember(processed)
		return nil
	})
	if err != nil {
		return err
	}
	mlx.Eval(r.state.Arrays()...)
	return nil
}

// first starts the computation of the first token from the last tokens of
// the prompt.
func (r *run) first(prompt []int32) (*mlx.Array, error) {
	return r.sample(func() *mlx.Array { return mlx.FromInt32(prompt, len(prompt)) })
}

// step starts the computation of the token after y, which need not be
// evaluated yet.
func (r *run) step(y *mlx.Array) (*mlx.Array, error) {
	return r.sample(func() *mlx.Array { return y })
}

// sample feeds the one-dimensional tokens that input returns to the model,
// samples the next token from the logits of the last position and starts its
// evaluation. The token is an int32 array of shape [1], which the caller
// owns.
func (r *run) sample(input func() *mlx.Array) (*mlx.Array, error) {
	var y *mlx.Array
	err := mlx.Scoped(func() (err error) {
		inputTokens := input()
		logits := r.model.Forward(mlx.ExpandDims(inputTokens, 0), r.state)
		length, vocab := logits.Dim(1), logits.Dim(2)
		logits = mlx.Reshape(mlx.Slice(logits, []int{0, length - 1, 0}, []int{1, length, vocab}), 1, vocab)

		r.remember(inputTokens)
		sampled, err := r.sampler.Sample(r.tokens, logits)
		if err != nil {
			return err
		}
		// Converting inside the graph spares a second evaluation per token
		// just to read it.
		y = mlx.AsType(sampled, mlx.Int32)
		mlx.Keep(y)
		mlx.Keep(r.state.Arrays()...)
		return nil
	})
	if err != nil {
		return nil, err
	}
	mlx.AsyncEval(y)
	return y, nil
}

// remember appends tokens to the history the penalties look at. It must run
// in a scope, out of which it keeps the history. The first tokens of a run
// become the history themselves: they are prompt tokens made in that scope,
// never a sampled token, which the caller owns.
func (r *run) remember(tokens *mlx.Array) {
	if !r.sampler.UsesTokens() {
		return
	}
	if r.tokens == nil {
		r.tokens = tokens
	} else {
		old := r.tokens
		r.tokens = mlx.Concatenate(0, old, tokens)
		old.Free()
	}
	mlx.Keep(r.tokens)
}
