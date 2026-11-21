# mlx-lm-go needs Go 1.27+ and Homebrew's mlx-c (brew install mlx-c).

.PHONY: build test test-race vet fixtures

build:
	go build -o bin/mlx-lm-go ./cmd/mlx-lm-go

test:
	go test ./...

test-race:
	go test -race ./...

vet:
	go vet ./...

# Regenerate golden fixtures from the Python reference.
# PYTHON records tokenizer, template and server behaviour and must have mlx-lm 0.31.3.
# MLX_PYTHON records model numerics and must run the MLX version this module links.
PYTHON ?= python3
MLX_PYTHON ?= /opt/homebrew/opt/mlx-lm/libexec/bin/python
fixtures:
	HF_HUB_OFFLINE=1 $(PYTHON) tools/oracle/make_text_fixtures.py
	$(PYTHON) tools/oracle/record_server.py
	HF_HUB_OFFLINE=1 $(MLX_PYTHON) tools/oracle/make_model_fixtures.py
