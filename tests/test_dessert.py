from _pytest.assertion.rewrite import AssertionRewritingHook as PytestRewriteHook
import os
import shutil
import sys
from contextlib import contextmanager
from tempfile import mkdtemp

import emport

import dessert
import pytest


def test_dessert(module):
    with pytest.raises(AssertionError) as error:
        module.func()

    assert 'dessert*' in str(error.value)
    assert "where" in str(error.value)
    assert "+" in str(error.value)

@pytest.fixture(scope='session', autouse=True)
def mark_dessert():
    assert not dessert.rewrite._MARK_ASSERTION_INTROSPECTION
    dessert.rewrite._MARK_ASSERTION_INTROSPECTION = True

@pytest.fixture
def module(request, source_filename, assertion_line):
    with dessert.rewrite_assertions_context():
        with _disable_pytest_rewriting():
            module = emport.import_file(source_filename)

    @request.addfinalizer
    def drop_from_sys_modules():
        sys.modules.pop(module.__name__)

    return module

@contextmanager
def _disable_pytest_rewriting():
    old_meta_path = sys.meta_path[:]
    try:
        for index, plugin in reversed(list(enumerate(sys.meta_path))):
            if isinstance(plugin, PytestRewriteHook):
                sys.meta_path.pop(index)
        yield
    finally:
        sys.meta_path[:] = old_meta_path


@pytest.fixture(params=[
    "assert x() + y()",
    "assert f(1) > g(100)",
    "assert f(g(2)) == f(g(1))",
    ])
def assertion_line(request):
    return request.param

@pytest.fixture
def source(assertion_line):
    returned = """def f(x):
    return x

x = lambda: 1
y = lambda: -1

g = h = f

def func():
    variable = False

    {0}
    """.format(assertion_line)
    return returned


@pytest.fixture
def source_filename(request, source):
    path = mkdtemp()

    @request.addfinalizer
    def delete():
        shutil.rmtree(path)

    filename = os.path.join(path, "sourcefile.py")
    with open(filename, "w") as f:
        f.write(source)

    return filename
