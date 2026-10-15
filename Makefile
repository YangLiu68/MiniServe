.PHONY: install lint test test-real serve serve-toy bench-toy bench-real profile docker

install:            ## dev environment (torch CPU on Linux, MPS-capable on macOS)
	uv sync --extra hf --extra bench

lint:
	uv run ruff check src tests bench examples && uv run ruff format --check src tests bench examples

test:               ## toy + tiny random models; no downloads
	uv run pytest -q

test-real:          ## downloads SmolLM2-135M (~270 MB)
	MINISERVE_RUN_REAL_MODEL_TESTS=1 uv run pytest -q -m real_model

serve:              ## native engine on SmolLM2 (auto device)
	uv run miniserve

serve-toy:
	MINISERVE_BACKEND=toy MINISERVE_TOY_STEP_OVERHEAD_MS=20 uv run miniserve

bench-toy:
	uv run miniserve-bench suite bench/suites/toy-batching.json --out results/toy-batching/latest
	uv run miniserve-bench report results/toy-batching/latest

profile:
	uv run python bench/profile_step.py --device auto --dtype auto

docker:
	docker build -t miniserve .
