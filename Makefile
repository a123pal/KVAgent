.PHONY: install test lint bench analyze

install:
	python -m pip install -e '.[dev]'

test:
	pytest -q

lint:
	ruff check src tests scripts

bench:
	kvagent-bench --artifact examples/reverse_lines.c --output results/benchmark.json

analyze:
	kvagent-analyze results/benchmark.json
