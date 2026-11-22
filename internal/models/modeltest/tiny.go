package modeltest

import (
	"encoding/binary"
	"encoding/json"
	"hash/fnv"
	"maps"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// The dimensions of the tiny model: small enough to write to disk in every
// test, large enough for 4-bit quantization in groups of 64.
const (
	TinyHiddenSize       = 64
	TinyIntermediateSize = 128
	TinyVocabSize        = 32
	TinyEOS              = 31
)

// Tensor is one tensor of a safetensors file.
type Tensor struct {
	DType string // as safetensors names it: "F32", "F16", "U32", "U8"
	Shape []int
	Data  []byte // little endian, row-major; nil means zeros
}

// itemSize returns the bytes of one element.
func (t Tensor) itemSize() int {
	switch t.DType {
	case "U8":
		return 1
	case "F16":
		return 2
	default:
		return 4
	}
}

// size returns the bytes of the tensor's data.
func (t Tensor) size() int {
	n := t.itemSize()
	for _, d := range t.Shape {
		n *= d
	}
	return n
}

// Zeros returns a tensor of zeros.
func Zeros(dtype string, shape ...int) Tensor {
	return Tensor{DType: dtype, Shape: shape}
}

// source returns a random source that depends only on name, so that a tensor
// has the same content in every run and whatever else the model holds.
func source(name string) *rand.Rand {
	h := fnv.New64a()
	h.Write([]byte(name))
	return rand.New(rand.NewPCG(h.Sum64(), 20261121))
}

// Random returns a float32 tensor of normal values with the given standard
// deviation, drawn from a source seeded with name.
func Random(name string, scale float64, shape ...int) Tensor {
	t := Tensor{DType: "F32", Shape: shape}
	r := source(name)
	t.Data = make([]byte, t.size())
	for i := 0; i < len(t.Data); i += 4 {
		binary.LittleEndian.PutUint32(t.Data[i:], math.Float32bits(float32(r.NormFloat64()*scale)))
	}
	return t
}

// RandomBits returns a uint32 tensor of random bits, drawn from a source
// seeded with name: the packed weights of a quantized layer.
func RandomBits(name string, shape ...int) Tensor {
	t := Tensor{DType: "U32", Shape: shape}
	r := source(name)
	t.Data = make([]byte, t.size())
	for i := 0; i < len(t.Data); i += 4 {
		binary.LittleEndian.PutUint32(t.Data[i:], r.Uint32())
	}
	return t
}

// TinyConfig returns the config.json of the tiny model as a map, for tests
// to change.
func TinyConfig() map[string]any {
	return map[string]any{
		"model_type":          "qwen2",
		"hidden_size":         TinyHiddenSize,
		"num_hidden_layers":   2,
		"intermediate_size":   TinyIntermediateSize,
		"num_attention_heads": 4,
		"num_key_value_heads": 2,
		"rms_norm_eps":        1e-6,
		"vocab_size":          TinyVocabSize,
		"eos_token_id":        TinyEOS,
	}
}

// TinyQuantizedConfig returns the config.json of the tiny model with its
// matrices quantized as TinyQuantized makes them. The reference quantizes a
// model only when config.json says it is.
func TinyQuantizedConfig() map[string]any {
	config := TinyConfig()
	config["quantization"] = map[string]any{"group_size": 64, "bits": 4}
	return config
}

// linear describes a linear layer of the tiny model.
type linear struct {
	out, in int
	bias    bool
}

// tinyLinears lists the linear layers of the tiny model by path.
func tinyLinears() map[string]linear {
	const h, m = TinyHiddenSize, TinyIntermediateSize
	linears := map[string]linear{}
	for _, layer := range []string{"model.layers.0.", "model.layers.1."} {
		linears[layer+"self_attn.q_proj"] = linear{h, h, true}
		linears[layer+"self_attn.k_proj"] = linear{h / 2, h, true}
		linears[layer+"self_attn.v_proj"] = linear{h / 2, h, true}
		linears[layer+"self_attn.o_proj"] = linear{h, h, false}
		linears[layer+"mlp.gate_proj"] = linear{m, h, false}
		linears[layer+"mlp.up_proj"] = linear{m, h, false}
		linears[layer+"mlp.down_proj"] = linear{h, m, false}
	}
	return linears
}

// tinyNorms returns the tensors of the norms of the tiny model.
func tinyNorms() map[string]Tensor {
	tensors := map[string]Tensor{}
	for _, name := range []string{
		"model.layers.0.input_layernorm", "model.layers.0.post_attention_layernorm",
		"model.layers.1.input_layernorm", "model.layers.1.post_attention_layernorm",
		"model.norm",
	} {
		name += ".weight"
		tensors[name] = Random(name, 0.5, TinyHiddenSize)
	}
	return tensors
}

// TinyDense returns the tensors of the tiny model with dense float32
// weights: two layers of random values, the same in every run.
func TinyDense() map[string]Tensor {
	tensors := tinyNorms()
	tensors["model.embed_tokens.weight"] = Random("model.embed_tokens.weight", 0.5, TinyVocabSize, TinyHiddenSize)
	for path, l := range tinyLinears() {
		tensors[path+".weight"] = Random(path+".weight", 0.2, l.out, l.in)
		if l.bias {
			tensors[path+".bias"] = Random(path+".bias", 0.2, l.out)
		}
	}
	return tensors
}

// TinyQuantized returns the tensors of the tiny model with every matrix
// quantized to 4 bits in groups of 64, the layout of the reference model.
// Any bits are valid packed weights, so these are random too.
func TinyQuantized() map[string]Tensor {
	tensors := tinyNorms()
	quantize := func(path string, out, in int) {
		tensors[path+".weight"] = RandomBits(path+".weight", out, in*4/32)
		tensors[path+".scales"] = Random(path+".scales", 0.02, out, in/64)
		tensors[path+".biases"] = Random(path+".biases", 0.1, out, in/64)
	}
	quantize("model.embed_tokens", TinyVocabSize, TinyHiddenSize)
	for path, l := range tinyLinears() {
		quantize(path, l.out, l.in)
		if l.bias {
			tensors[path+".bias"] = Random(path+".bias", 0.2, l.out)
		}
	}
	return tensors
}

// WriteModel writes a model directory with the given config.json and a
// model.safetensors holding the tensors, and returns it. The directory is
// removed when the test ends.
func WriteModel(t testing.TB, config map[string]any, tensors map[string]Tensor) string {
	t.Helper()
	dir := t.TempDir()
	raw, err := json.Marshal(config)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), raw, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), Safetensors(t, tensors), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

// Safetensors returns the content of a safetensors file with the given
// tensors: the length of a JSON header, the header, then the data of the
// tensors in the order of their names.
func Safetensors(t testing.TB, tensors map[string]Tensor) []byte {
	t.Helper()
	type entry struct {
		DType   string `json:"dtype"`
		Shape   []int  `json:"shape"`
		Offsets [2]int `json:"data_offsets"`
	}
	header := map[string]entry{}
	var data []byte
	for _, name := range slices.Sorted(maps.Keys(tensors)) {
		tensor := tensors[name]
		content := tensor.Data
		if content == nil {
			content = make([]byte, tensor.size())
		}
		if len(content) != tensor.size() {
			t.Fatalf("tensor %s has %d bytes of data, its shape %v holds %d", name, len(content), tensor.Shape, tensor.size())
		}
		shape := tensor.Shape
		if shape == nil {
			shape = []int{}
		}
		header[name] = entry{DType: tensor.DType, Shape: shape, Offsets: [2]int{len(data), len(data) + len(content)}}
		data = append(data, content...)
	}
	raw, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	for len(raw)%8 != 0 {
		raw = append(raw, ' ')
	}
	file := binary.LittleEndian.AppendUint64(nil, uint64(len(raw)))
	file = append(file, raw...)
	return append(file, data...)
}
