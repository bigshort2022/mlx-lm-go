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

# Regenerate golden fixtures from the Python reference. PYTHON must have mlx-lm 0.31.3.
PYTHON ?= python3
fixtures:
	HF_HUB_OFFLINE=1 $(PYTHON) tools/oracle/make_text_fixtures.py
	$(PYTHON) tools/oracle/record_server.py
