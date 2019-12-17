# pylint: disable=unused-import,import-error
import fnmatch
import os
import sys
from os.path import sep
from posixpath import sep as posix_sep


if sys.version_info[:2] >= (3, 6):
    from pathlib import Path, PurePath
else:
    from pathlib2 import Path, PurePath


def fnmatch_ex(pattern, path):
    """FNMatcher port from py.path.common which works with PurePath() instances.

    The difference between this algorithm and PurePath.match() is that the latter matches "**" glob expressions
    for each part of the path, while this algorithm uses the whole path instead.

    For example:
        "tests/foo/bar/doc/test_foo.py" matches pattern "tests/**/doc/test*.py" with this algorithm, but not with
        PurePath.match().

    This algorithm was ported to keep backward-compatibility with existing settings which assume paths match according
    this logic.

    References:
    * https://bugs.python.org/issue29249
    * https://bugs.python.org/issue34731
    """

    path = PurePath(path)
    iswin32 = sys.platform.startswith("win")

    if iswin32 and sep not in pattern and posix_sep in pattern:
        # Running on Windows, the pattern has no Windows path separators,
        # and the pattern has one or more Posix path separators. Replace
        # the Posix path separators with the Windows path separator.
        pattern = pattern.replace(posix_sep, sep)

    if sep not in pattern:
        name = path.name
    else:
        name = str(path)
        if path.is_absolute() and not os.path.isabs(pattern):
            pattern = "*{}{}".format(os.sep, pattern)
    return fnmatch.fnmatch(name, pattern)
