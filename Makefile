default: test

test: env
	.env/bin/py.test -x tests

env: .env/.up-to-date


.env/.up-to-date: setup.py
	virtualenv --no-site-packages .env
	.env/bin/pip install -e .
	.env/bin/pip install -r ./*.egg-info/requires.txt
	.env/bin/pip install pytest
	touch $@

