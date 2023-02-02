default: test

test: env
	.env/bin/pytest -x tests

env: .env/.up-to-date


.env/.up-to-date: setup.py Makefile setup.cfg
	virtualenv --no-site-packages .env
	.env/bin/pip install -e '.[testig]'
	.env/bin/pip install -r ./*.egg-info/requires.txt || true
	touch $@

