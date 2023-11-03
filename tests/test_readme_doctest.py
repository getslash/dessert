import doctest
import os
import shutil

import pytest
import tempfile


def test_readme_doctests(tmp_filename):
    readme_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "README.md"))
    assert os.path.exists(readme_path)
    result = doctest.testfile(readme_path, module_relative=False, globs={"tmp_filename": tmp_filename})
    assert result.failed == 0

@pytest.fixture(name="tmp_filename")
def tmp_filename_fx(request):
    tmp_dir = tempfile.mkdtemp()
    @request.addfinalizer
    def delete():
        shutil.rmtree(tmp_dir)
    return os.path.join(tmp_dir, "filename.py")
