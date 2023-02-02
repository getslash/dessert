default: test

test: env
	.env/bin/pytest -x tests

env: .env/.up-to-date


.env/.up-to-date: pyproject.toml Makefile
	python3 -m venv .env
	.env/bin/pip install -e '.[testing]'
	touch $@

