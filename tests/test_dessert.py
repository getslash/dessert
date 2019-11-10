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


def test_disable_introspection(add_assert_message, module, assert_message):
    with _disable_introspection():
        with pytest.raises(AssertionError) as error:
            module.func()
        if not add_assert_message:
            assert 'dessert*' in str(error.value)
            assert "where" in str(error.value)
            assert "+" in str(error.value)
        else:
            assert assert_message in str(error.value)
            assert "+" not in str(error.value)


def test_warnings_from_rewrite(source_filename):
    tmp_dir = os.path.dirname(source_filename)
    full_path = os.path.join(tmp_dir, 'file_with_warnings.py')
    with open(full_path, "w") as f:
        f.write(r"""
import warnings
warnings.simplefilter('always')
warnings.warn('Some import warning')

def func():
    assert True
""")
    with dessert.rewrite_assertions_context():
        with _disable_pytest_rewriting():
            with pytest.warns(None) as caught:
                emport.import_file(full_path)
            [warning] = caught.list
            assert warning.filename == full_path


@pytest.fixture(scope='session', autouse=True)
def mark_dessert():
    # pylint: disable=protected-access
    assert not dessert.rewrite._MARK_ASSERTION_INTROSPECTION
    dessert.rewrite._MARK_ASSERTION_INTROSPECTION = True

@pytest.fixture
def module(request, source_filename):
    with dessert.rewrite_assertions_context():
        with _disable_pytest_rewriting():
            module = emport.import_file(source_filename)

    @request.addfinalizer
    def drop_from_sys_modules():  # pylint: disable=unused-variable
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

@contextmanager
def _disable_introspection():
    dessert.disable_message_introspection()
    try:
        yield
    finally:
        dessert.enable_message_introspection()


@pytest.fixture(params=[
    "assert x() + y()",
    "assert f(1) > g(100)",
    "assert f(g(2)) == f(g(1))",
    ])
def assertion_line(request):
    return request.param


@pytest.fixture(params=[True, False])
def add_assert_message(request):
    return request.param

@pytest.fixture
def assert_message(request):
    return 'msg'


@pytest.fixture
def source(assertion_line, add_assert_message, assert_message):
    if add_assert_message:
        assertion_line += ", '{}'".format(assert_message)
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
    def delete():  # pylint: disable=unused-variable
        shutil.rmtree(path)

    filename = os.path.join(path, "sourcefile.py")
    with open(filename, "w") as f:
        f.write(source)

    return filename
