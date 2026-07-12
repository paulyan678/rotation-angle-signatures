.PHONY: install test lint smoke manifests figures

install:
	python -m pip install -e '.[dev,medical]'

test:
	pytest

lint:
	ruff check src tests

smoke:
	rotation-patterns smoke --config configs/smoke.yaml

manifests:
	rotation-patterns make-manifests --config configs/paper.yaml

figures:
	rotation-patterns figures --config configs/paper.yaml

