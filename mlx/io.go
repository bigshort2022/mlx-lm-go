package mlx

/*
#include <stdlib.h>
#include <mlx/c/mlx.h>
*/
import "C"

import "unsafe"

// The statuses of mlx-c's map iterators other than failure.
const (
	iteratorHasEntry = 0
	iteratorDone     = 2
)

// LoadSafetensors reads the tensors and the string metadata of a safetensors
// file. The returned arrays belong to the caller, not to any scope. They are
// lazy: MLX reads a tensor's data from the file when it is first evaluated,
// which must happen on the thread that loaded it.
func LoadSafetensors(path string) (map[string]*Array, map[string]string, error) {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	tensors := C.mlx_map_string_to_array_new()
	defer C.mlx_map_string_to_array_free(tensors)
	entries := C.mlx_map_string_to_string_new()
	defer C.mlx_map_string_to_string_free(entries)
	// Reading a file is the one operation MLX implements only for the CPU.
	cpu := C.mlx_default_cpu_stream_new()
	defer C.mlx_stream_free(cpu)

	if C.mlx_load_safetensors(&tensors, &entries, cpath, cpu) != 0 {
		return nil, nil, takeError()
	}
	metadata, err := goStrings(entries)
	if err != nil {
		return nil, nil, err
	}
	arrays, err := goArrays(tensors)
	if err != nil {
		return nil, nil, err
	}
	return arrays, metadata, nil
}

// goArrays returns the arrays of an MLX map as caller-owned Arrays.
func goArrays(m C.mlx_map_string_to_array) (map[string]*Array, error) {
	it := C.mlx_map_string_to_array_iterator_new(m)
	defer C.mlx_map_string_to_array_iterator_free(it)
	arrays := make(map[string]*Array)
	for {
		var key *C.char
		var value C.mlx_array
		switch C.mlx_map_string_to_array_iterator_next(&key, &value, it) {
		case iteratorHasEntry:
			arrays[C.GoString(key)] = &Array{c: value}
		case iteratorDone:
			return arrays, nil
		default:
			for _, a := range arrays {
				a.Free()
			}
			return nil, takeError()
		}
	}
}

// goStrings returns the entries of an MLX map as a Go map.
func goStrings(m C.mlx_map_string_to_string) (map[string]string, error) {
	it := C.mlx_map_string_to_string_iterator_new(m)
	defer C.mlx_map_string_to_string_iterator_free(it)
	entries := make(map[string]string)
	for {
		var key, value *C.char
		switch C.mlx_map_string_to_string_iterator_next(&key, &value, it) {
		case iteratorHasEntry:
			entries[C.GoString(key)] = C.GoString(value)
		case iteratorDone:
			return entries, nil
		default:
			return nil, takeError()
		}
	}
}
