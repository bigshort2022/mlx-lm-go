# M1: `mlx` bindings and the inference thread

> **For agentic workers:** implement task by task with red/green TDD. For every test: write it, run it, see it fail for the stated reason, then write the smallest code that passes. Steps use checkbox (`- [ ]`) syntax.

**Goal:** a Go package `mlx` that exposes the part of mlx-c needed to run a quantized transformer, and a package `internal/exec` that serialises work onto one OS thread.

**Architecture:** `mlx` is the only package that imports `"C"`. Arrays are C handles owned by scopes. Operations panic with `*mlx.Error`; `mlx.Scoped` turns that panic into a returned error. I/O functions return errors directly.

**Tech stack:** Go 1.27, cgo, Homebrew `mlx-c` 0.8.0 (headers in `/opt/homebrew/include/mlx/c`, MLX 0.32.4).

---

## Starting point

- `mlx/cgo.go` holds the `#cgo` flags. `mlx/version.go` and `mlx/version_test.go` show the binding style and already pass.
- Read the C API in `/opt/homebrew/include/mlx/c/{array,ops,fast,io,map,vector,string,stream,device,memory,error,random,transforms,optional}.h`. Small C examples are in `/Users/b33j_02/sandbox/mlx-c/examples/`.
- For expected values use the Python reference at `/Users/b33j_02/sandbox/mlx/.venv/bin/python` (has `numpy` and `mlx.core`). Compute the value there, then hard-code it in the Go test.

## Files you own

Create: `mlx/{error,scope,stream,dtype,array,ops,fast,random,io,memory}.go`, a `_test.go` beside each, `mlx/testdata/small.safetensors`, `internal/exec/exec.go`, `internal/exec/exec_test.go`.
Do not touch anything else, including `go.mod`, `go.sum` and `mlx/cgo.go` (ask if the link flags must change).

## Contract

Later milestones are written against these signatures. Keep them exactly.

```go
package mlx

// Error is an error reported by MLX.
type Error struct{ Msg string }
func (e *Error) Error() string

// DType mirrors mlx_dtype, in the same order.
type DType int
const (
	Bool DType = iota
	Uint8; Uint16; Uint32; Uint64
	Int8; Int16; Int32; Int64
	Float16; Float32; Float64; BFloat16; Complex64
)

type Array struct{ /* unexported */ }

// Lifetime.
func Scoped(fn func() error) error // frees every Array created inside fn, except kept ones
func Keep(arrs ...*Array)          // moves arrays to the enclosing scope; at the outermost scope the caller owns them
func (a *Array) Free()             // safe to call twice
func (a *Array) Valid() bool       // false once freed

// Construction.
func FromFloat32(data []float32, shape ...int) *Array
func FromInt32(data []int32, shape ...int) *Array
func Float(v float32) *Array
func Int(v int32) *Array
func Zeros(dtype DType, shape ...int) *Array

// Inspection. Float32s and Int32s evaluate the array and require that exact dtype.
func (a *Array) Shape() []int
func (a *Array) NDim() int
func (a *Array) Dim(axis int) int // negative axes count from the end
func (a *Array) Size() int
func (a *Array) DType() DType
func (a *Array) Float32s() []float32
func (a *Array) Int32s() []int32
func Eval(arrs ...*Array)
func AsyncEval(arrs ...*Array)

// Operations.
func Add(a, b *Array) *Array
func Subtract(a, b *Array) *Array
func Multiply(a, b *Array) *Array
func Divide(a, b *Array) *Array
func Negative(a *Array) *Array
func Exp(a *Array) *Array
func Log(a *Array) *Array
func Sigmoid(a *Array) *Array
func Maximum(a, b *Array) *Array
func Minimum(a, b *Array) *Array
func Greater(a, b *Array) *Array
func Less(a, b *Array) *Array
func Where(cond, x, y *Array) *Array
func Matmul(a, b *Array) *Array
func Reshape(a *Array, shape ...int) *Array
func Transpose(a *Array, axes ...int) *Array
func ExpandDims(a *Array, axis int) *Array
func Squeeze(a *Array, axis int) *Array
func AsType(a *Array, dtype DType) *Array
func Concatenate(axis int, arrs ...*Array) *Array
func Slice(a *Array, start, stop []int) *Array // one entry per dimension, stride 1
func Take(a, indices *Array, axis int) *Array
func TakeAlongAxis(a, indices *Array, axis int) *Array
func PutAlongAxis(a, indices, values *Array, axis int) *Array
func Argmax(a *Array, axis int, keepDims bool) *Array
func Softmax(a *Array, axis int) *Array
func LogSumExp(a *Array, axis int, keepDims bool) *Array
func Cumsum(a *Array, axis int) *Array
func Sort(a *Array, axis int) *Array
func Argsort(a *Array, axis int) *Array
func ArgPartition(a *Array, kth, axis int) *Array
func QuantizedMatmul(x, w, scales, biases *Array, transpose bool, groupSize, bits int, mode string) *Array
func Dequantize(w, scales, biases *Array, groupSize, bits int, mode string) *Array

// Fast ops.
func RMSNorm(x, weight *Array, eps float32) *Array
func RoPE(x *Array, dims int, traditional bool, base, scale float32, offset int) *Array
func ScaledDotProductAttention(q, k, v *Array, scale float32, maskMode string, mask *Array) *Array // mask may be nil

// Random.
func Seed(seed uint64)
func Categorical(logits *Array, axis int) *Array

// I/O. The returned arrays belong to the caller, not to any scope.
func LoadSafetensors(path string) (map[string]*Array, map[string]string, error)

// Memory, in bytes.
func ActiveMemory() uint64
func PeakMemory() uint64
func CacheMemory() uint64
func ResetPeakMemory()
func ClearCache()
func SetCacheLimit(limit uint64) (previous uint64)
func SetMemoryLimit(limit uint64) (previous uint64)
func SetWiredLimit(limit uint64) (previous uint64)
```

```go
package exec

// Thread runs functions one at a time on a single locked OS thread.
type Thread struct{ /* unexported */ }
func Start() *Thread
// Do queues fn and waits for it. Functions run in arrival order. If ctx ends
// before fn starts, fn never runs and Do returns ctx.Err(). Once started, fn
// runs to completion. A panic in fn is returned as an error and the thread
// keeps serving.
func (t *Thread) Do(ctx context.Context, fn func() error) error
// Close stops the thread after the running function. Later Do calls return an error.
func (t *Thread) Close()
```

Rules that hold for every wrapper:

- All operations run on the default GPU stream.
- A failed MLX call must not exit the process. mlx-c's default error handler calls `exit`; install your own with `mlx_set_error_handler` when the package loads.
- No finalizers. Lifetime is scopes and `Free` only.
- The package is single-threaded by contract. Tests must not use `t.Parallel()`.

## Tasks

Run tests with `go test ./mlx/ -run <Name> -v`. Never run `go build ./...`, `go test ./...`, `go mod tidy` or `go get`.

### Task 1: errors do not kill the process

- [ ] Test `TestShapeMismatchReturnsErrorFromScoped`: inside `Scoped`, `Matmul` of shapes `[2,3]` and `[2,3]`, then `Eval`. Expect `Scoped` to return a non-nil error that `errors.As` to `*Error` with a non-empty `Msg`. The test process must survive. This needs the minimum of `FromFloat32`, `Matmul`, `Eval`, `Scoped`; write only that much.
- [ ] Test `TestScopedPropagatesOtherPanics`: `Scoped(func() error { panic("boom") })` panics with `"boom"`.
- [ ] Test `TestScopedReturnsFnError`: returns the same error value fn returned.

### Task 2: scopes own arrays

- [ ] `TestScopedFreesArraysCreatedInside`: capture two arrays created in a scope; after `Scoped` returns, `Valid()` is false for both.
- [ ] `TestKeepMovesArrayToOuterScope`: nested scopes; an array kept in the inner scope is valid after the inner scope and invalid after the outer one.
- [ ] `TestKeepAtOutermostScopeLeavesOwnershipWithCaller`: kept array is valid after `Scoped`; `Free` makes it invalid; a second `Free` does not crash.
- [ ] `TestArraysCreatedOutsideAnyScopeAreCallerOwned`: created at top level, stays valid until `Free`.
- [ ] `TestScopedMemoryReturnsToBaseline`: record `ActiveMemory()`; run 1,000 iterations of a scope that creates and evaluates a `[256,256]` float array sum; `ClearCache()`; active memory is within 1 MiB of the baseline.

### Task 3: construction and inspection

- [ ] Round trips: `FromFloat32` and `FromInt32` with shape `[2,3]` return the same data, `Shape`, `NDim`, `Size`, `DType`. `Dim(-1)` equals the last dimension.
- [ ] `Float(1.5)` and `Int(7)` are zero-dimensional with the right dtype and value.
- [ ] `Zeros(Float32, 2, 2)` is four zeros.
- [ ] `FromFloat32` with data length not matching the shape panics `*Error` (check it yourself; do not call MLX with a bad length).
- [ ] `Float32s()` on an `Int32` array panics `*Error`.

### Task 4: operations

One test per function, each with an expected value computed in Python beforehand. Minimum cases:

- [ ] Element-wise ops on `[1,2,3]` and `[4,5,6]`; broadcasting a scalar.
- [ ] `Matmul` `[2,3]x[3,2]`.
- [ ] `Reshape`, `Transpose` with explicit axes on a `[2,3,4]` array, `ExpandDims`, `Squeeze`, `AsType` float to int.
- [ ] `Concatenate` on axis 0 and axis -1; `Slice` of a `[4,5]` array to rows 1..3, columns 2..5.
- [ ] `Take` on axis 0 with repeated indices; `TakeAlongAxis`; `PutAlongAxis`.
- [ ] `Argmax` with and without `keepDims`; `Softmax` sums to 1 and matches Python to 1e-6; `LogSumExp`; `Cumsum`.
- [ ] `Sort`, `Argsort`; `ArgPartition`: the `kth` element is in its sorted position.
- [ ] `Greater`, `Less`, `Where`.
- [ ] `QuantizedMatmul` and `Dequantize`: in Python, quantize a fixed `[8,64]` float matrix with `mx.quantize(w, group_size=64, bits=4)`, save `w_q`, `scales`, `biases`, an input `x` and the expected `mx.quantized_matmul(x, w_q, scales, biases, transpose=True, group_size=64, bits=4)` and `mx.dequantize(...)` to `mlx/testdata/small.safetensors`. The Go test loads the file and compares within 1e-4.

### Task 5: fast ops and random

Expected values again from Python `mlx.core.fast`, stored in `small.safetensors`.

- [ ] `RMSNorm` on a `[2,4]` input with a weight vector.
- [ ] `RoPE` on a `[1,2,4,8]` input with `dims=8`, `base=10000`, `scale=1`, at offset 0 and at offset 3.
- [ ] `ScaledDotProductAttention` with `maskMode` `""` and `"causal"` on `[1,2,4,8]` inputs.
- [ ] `Categorical`: after `Seed(7)`, sampling twice from the same logits gives the same pair as after a second `Seed(7)`. Logits with one finite entry always return its index.

### Task 6: safetensors

- [ ] `LoadSafetensors` on `small.safetensors` returns every key written by Python with the right shape and dtype, and the metadata map contains a key you wrote.
- [ ] Loading a missing path returns an error and does not exit or panic.
- [ ] Loaded arrays stay valid after an enclosing `Scoped` returns.

### Task 7: memory

- [ ] `ActiveMemory` grows after evaluating a 16 MiB array and falls after `Free` and `ClearCache`.
- [ ] `PeakMemory` is at least `ActiveMemory`; `ResetPeakMemory` lowers it.
- [ ] `SetCacheLimit(0)` returns the previous limit; restoring it returns 0.

### Task 8: the inference thread

Run with `go test -race ./internal/exec/ -v`.

- [ ] `TestDoRunsOneAtATime`: 100 goroutines call `Do`; a counter incremented on entry and decremented on exit never exceeds 1; all 100 return nil.
- [ ] `TestDoRunsInArrivalOrder`: queue 10 functions from one goroutine while the thread is blocked on a first function; they run in order.
- [ ] `TestDoSkipsFunctionWhenContextEndsWhileQueued`: block the thread, queue a second call, cancel its context, unblock; the second function never ran and `Do` returned `context.Canceled`.
- [ ] `TestDoReturnsPanicAsError`: a panicking function yields an error containing the panic value; a following `Do` still works.
- [ ] `TestDoAfterCloseReturnsError`.

## Done when

`go test ./mlx/ && go test -race ./internal/exec/ && go vet ./mlx/ ./internal/exec/` passes with clean output, and `gofmt -l mlx internal/exec` prints nothing.

## Report

For each task: the test names, the failure you saw before implementing (one line each), and the final pass. List any signature you could not keep and why. Do not commit.
