// Package hub locates model directories on disk. It never downloads.
package hub

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Resolve returns the directory holding a model's files.
//
// model is either an existing directory, returned unchanged, or a Hugging Face
// repository id such as "mlx-community/Qwen2.5-1.5B-Instruct-4bit", looked up
// in the local Hugging Face cache. The cache root is $HF_HUB_CACHE, else
// $HF_HOME/hub, else ~/.cache/huggingface/hub.
func Resolve(model string) (string, error) {
	if info, err := os.Stat(model); err == nil && info.IsDir() {
		return model, nil
	}

	root, err := cacheRoot()
	if err != nil {
		return "", err
	}
	repo := filepath.Join(root, "models--"+strings.ReplaceAll(model, "/", "--"))
	snapshots := filepath.Join(repo, "snapshots")

	if ref, err := os.ReadFile(filepath.Join(repo, "refs", "main")); err == nil {
		dir := filepath.Join(snapshots, strings.TrimSpace(string(ref)))
		if info, err := os.Stat(dir); err == nil && info.IsDir() {
			return dir, nil
		}
	}

	entries, err := os.ReadDir(snapshots)
	if err != nil || len(entries) == 0 {
		return "", fmt.Errorf("model %q is not a directory and is not in the Hugging Face cache at %s; run: hf download %s", model, root, model)
	}
	if len(entries) > 1 {
		return "", fmt.Errorf("model %q has %d cached snapshots and no refs/main in %s; pass the snapshot directory instead", model, len(entries), repo)
	}
	return filepath.Join(snapshots, entries[0].Name()), nil
}

func cacheRoot() (string, error) {
	if dir := os.Getenv("HF_HUB_CACHE"); dir != "" {
		return dir, nil
	}
	if home := os.Getenv("HF_HOME"); home != "" {
		return filepath.Join(home, "hub"), nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("locating the Hugging Face cache: %w", err)
	}
	return filepath.Join(home, ".cache", "huggingface", "hub"), nil
}
