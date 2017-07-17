import sys

from contextlib import contextmanager
from .conf import conf, DISABLE_RETROSPECTION_KEY
from .rewrite import AssertionRewritingHook


@contextmanager
def rewrite_assertions_context():
    hook = AssertionRewritingHook()
    prev_meta_path = sys.meta_path[:]
    sys.meta_path.insert(0, hook)
    try:
        yield
    finally:
        sys.meta_path[:] = prev_meta_path


def disable_retrospection():
    conf[DISABLE_RETROSPECTION_KEY] = True

def enable_retrospection():
    conf[DISABLE_RETROSPECTION_KEY] = False
