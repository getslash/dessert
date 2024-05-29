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
            with pytest.warns((UserWarning)) as caught:
                emport.import_file(full_path)
            # caught.list is list of warnings.WarningMessage
            # The message member is the actual warning
            [warning] = [
                x
                for x in caught.list
                if isinstance(x.message, UserWarning)
            ]
            assert warning.filename == full_path


@pytest.fixture(scope='session', autouse=True)
def mark_dessert():
    # pylint: disable=protected-access
    assert not dessert.rewrite._MARK_ASSERTION_INTROSPECTION
    dessert.rewrite._MARK_ASSERTION_INTROSPECTION = True

@pytest.fixture(name="module")
def module_fx(request, source_filename):
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


@pytest.fixture(name="assertion_line", params=[
    "assert x() + y()",
    "assert f(1) > g(100)",
    "assert f(g(2)) == f(g(1))",
    ])
def assertion_line_fx(request):
    return request.param


@pytest.fixture(name="add_assert_message", params=[True, False])
def add_assert_message_fx(request):
    return request.param

@pytest.fixture(name="assert_message")
def assert_message_fx():
    return 'msg'


@pytest.fixture(name="source")
def source_fx(assertion_line, add_assert_message, assert_message):
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


@pytest.fixture(name="source_filename")
def source_filename_fx(request, source):
    path = mkdtemp()

    @request.addfinalizer
    def delete():  # pylint: disable=unused-variable
        shutil.rmtree(path)

    filename = os.path.join(path, "sourcefile.py")
    with open(filename, "w") as f:
        f.write(source)

    return filename


@pytest.mark.skipif(
    sys.version_info < (3, 9),
    reason="importlib.resources.files was introduced in 3.9",
)
def test_load_resource_via_files_with_rewrite() -> None:
    package_dir = mkdtemp()

    with open(os.path.join(package_dir, '__init__.py'), 'w') as init_file:
        init_file.write("""from importlib.resources import files
assert files(__package__).exists()""")

    with dessert.rewrite_assertions_context():
        emport.import_file(os.path.join(package_dir, '__init__.py'))
