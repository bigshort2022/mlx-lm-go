# mlx-lm-go needs Go 1.27+ and Homebrew's mlx-c (brew install mlx-c).

.PHONY: build test test-race vet fixtures

VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)

build:
	go build -ldflags "-X main.version=$(VERSION)" -o bin/mlx-lm-go ./cmd/mlx-lm-go

test:
	go test ./...

test-race:
	go test -race ./...

vet:
	go vet ./...

# Regenerate golden fixtures from the Python reference.
# PYTHON records tokenizer and chat-template behaviour (transformers, tokenizers, jinja2).
# MLX_PYTHON records server exchanges and model numerics; it must have mlx-lm 0.32.2
# and run the MLX version this module links.
PYTHON ?= python3
MLX_PYTHON ?= /opt/homebrew/opt/mlx-lm/libexec/bin/python
fixtures:
	HF_HUB_OFFLINE=1 $(PYTHON) tools/oracle/make_text_fixtures.py
	HF_HUB_OFFLINE=1 $(PYTHON) tools/oracle/make_template_differential.py
	$(MLX_PYTHON) tools/oracle/record_server.py
	HF_HUB_OFFLINE=1 $(MLX_PYTHON) tools/oracle/make_model_fixtures.py
