import sys

from contextlib import contextmanager

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
