from __future__ import print_function

import os
import shutil
import functools
import pytest
import subprocess
import sys


def test_example_code_simple_assertion(run_example):
    assert run_example('no_message_simple_equality') == 'assert 1 == 2'


def test_example_message_assertion(run_example):
    assert run_example('with_message_simple_equality') == 'Here is an assertion message\nassert 1 == 2'



@pytest.fixture(name="run_example")
def run_example_fx(tmpdir):
    here = os.path.dirname(__file__)
    shutil.copy(os.path.join(here, 'driver.py'), str(tmpdir))
    shutil.copy(os.path.join(here, 'examples.py'), str(tmpdir))

    return functools.partial(_run, str(tmpdir.join('driver.py')))

def _run(filename, example_name):
    executable = sys.executable
    output = subprocess.check_output([executable, filename, example_name], stderr=subprocess.STDOUT).decode('utf-8')
    return output.strip()
