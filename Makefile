default: test

test: env
	.venv/bin/pytest -x tests

env:
	uv venv
	uv pip install -e ".[testing]"
