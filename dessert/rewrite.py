"""Rewrite assertion AST to produce nice error messages"""
# pylint: disable=protected-access,unused-import,logging-not-lazy,duplicate-string-formatting-argument,logging-format-interpolation,too-many-lines,unused-argument,broad-exception-caught
import ast
import errno
import functools
import importlib.abc
import importlib.machinery
import importlib.util
import io
import itertools
import marshal
import os
import struct
import sys
import tempfile
import tokenize
import types
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

import atomicwrites

from . import util
from .util import format_explanation as _format_explanation

from .pathlib import Path, PurePath, fnmatch_ex

import logging
from munch import Munch
from .__version__ import __version__ as version
from .saferepr import saferepr

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.ERROR)

_MARK_ASSERTION_INTROSPECTION = False


# pytest caches rewritten pycs in __pycache__.
PYTEST_TAG = "{}-pytest-{}".format(sys.implementation.cache_tag, version)
PYC_EXT = ".py" + (__debug__ and "c" or "o")
PYC_TAIL = "." + PYTEST_TAG + PYC_EXT


class AssertRewritingSession(object):

    def __init__(self):
        self._initialpaths = frozenset()

    def isinitpath(self, filename):
        return True

ast_Call = ast.Call


class AssertionRewritingHook(object):
    """PEP302 Import hook which rewrites asserts."""

    def __init__(self, config=None):
        self.config = config
        self.modules = {}
        self.session = AssertRewritingSession()
        self.state = Munch()
        self.fnpats = []
        self._rewritten_names: Dict[str, Path] = {}
        self._must_rewrite: Set[str] = set()
        # flag to guard against trying to rewrite a pyc file while we are already writing another pyc file,
        # which might result in infinite recursion (#3506)
        self._writing_pyc = False
        self._basenames_to_check_rewrite = {"conftest"}
        self._marked_for_rewrite_cache = {}  # type: Dict[str, bool]
        self._session_paths_checked = False

    def find_spec(self, name, path=None, target=None):
        if self._writing_pyc:
            return None
        state = self.state
        if self._early_rewrite_bailout(name, state):
            return None
        _logger.debug("find_spec called for %s (path: %s, target: %s)" % (name, path, target))

        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if (
            # the import machinery could not find a file to import
            spec is None
            # this is a namespace package (without `__init__.py`)
            # there's nothing to rewrite there
            # python3.5 - python3.6: `namespace`
            # python3.7+: `None`
            or spec.origin in {None, "namespace"}
            # we can only rewrite source files
            or not isinstance(spec.loader, importlib.machinery.SourceFileLoader)
            # if the file doesn't exist, we can't rewrite it
            or not os.path.exists(spec.origin)
        ):
            return None
        else:
            fn = spec.origin

        if not self._should_rewrite(name, fn, state):
            return None

        return importlib.util.spec_from_file_location(
            name,
            fn,
            loader=self,
            submodule_search_locations=spec.submodule_search_locations,
        )

    def create_module(self, spec):
        return None  # default behaviour is fine

    def exec_module(self, module):
        fn = module.__spec__.origin
        state = self.state

        self._rewritten_names[module.__name__] = fn

        # The requested module looks like a test file, so rewrite it. This is
        # the most magical part of the process: load the source, rewrite the
        # asserts, and load the rewritten source. We also cache the rewritten
        # module code in a special pyc. We must be aware of the possibility of
        # concurrent pytest processes rewriting and loading pycs. To avoid
        # tricky race conditions, we maintain the following invariant: The
        # cached pyc is always a complete, valid pyc. Operations on it must be
        # atomic. POSIX's atomic rename comes in handy.
        write = not sys.dont_write_bytecode
        cache_dir = os.path.join(os.path.dirname(fn), "__pycache__")
        if write:
            ok = try_mkdir(cache_dir)
            if not ok:
                write = False
                _logger.debug("read only directory: {}".format(os.path.dirname(fn)))

        cache_name = os.path.basename(fn)[:-3] + PYC_TAIL
        pyc = os.path.join(cache_dir, cache_name)
        # Notice that even if we're in a read-only directory, I'm going
        # to check for a cached pyc. This may not be optimal...
        co = _read_pyc(fn, pyc, _logger.debug)
        if co is None:
            _logger.debug("rewriting {!r}".format(fn))
            source_stat, co = _rewrite_test(fn, self.config)
            if write:
                self._writing_pyc = True
                try:
                    _write_pyc(state, co, source_stat, pyc)
                finally:
                    self._writing_pyc = False
        else:
            _logger.debug("found cached rewritten pyc for {!r}".format(fn))
        exec(co, module.__dict__)  # pylint: disable=exec-used

    def _early_rewrite_bailout(self, name, state):
        """This is a fast way to get out of rewriting modules. Profiling has
        shown that the call to PathFinder.find_spec (inside of the find_spec
        from this class) is a major slowdown, so, this method tries to
        filter what we're sure won't be rewritten before getting to it.
        """

        # Dessert couldn't use the pytest implementation for this method, because it removes
        # its reformatting
        return False

    def _should_rewrite(self, name, fn_pypath, state):
        return True

    def _is_marked_for_rewrite(self, name: str, state):
        try:
            return self._marked_for_rewrite_cache[name]
        except KeyError:
            for marked in self._must_rewrite:
                if name == marked or name.startswith(marked + "."):
                    _logger.debug(
                        "matched marked file {!r} (from {!r})".format(name, marked)
                    )
                    self._marked_for_rewrite_cache[name] = True
                    return True

            self._marked_for_rewrite_cache[name] = False
            return False

    def mark_rewrite(self, *names: str) -> None:
        """Mark import names as needing to be rewritten.

        The named module or package as well as any nested modules will
        be rewritten on import.
        """
        already_imported = (
            set(names).intersection(sys.modules).difference(self._rewritten_names)
        )
        for name in already_imported:
            mod = sys.modules[name]
            if not AssertionRewriter.is_rewrite_disabled(
                mod.__doc__ or ""
            ) and not isinstance(mod.__loader__, type(self)):
                self._warn_already_imported(name)
        self._must_rewrite.update(names)
        self._marked_for_rewrite_cache.clear()

    def _warn_already_imported(self, name):
        self.config.warn(
            'P1',
            'Module already imported so can not be re-written: %s' % name)

    def get_data(self, pathname):
        """Optional PEP302 get_data API.
        """
        with open(pathname, 'rb') as f:
            return f.read()

    if sys.version_info >= (3, 10):
        if sys.version_info >= (3, 12):
            from importlib.resources.abc import TraversableResources
        else:
            from importlib.abc import TraversableResources

        def get_resource_reader(self, name: str) -> TraversableResources:
            if sys.version_info < (3, 11):
                from importlib.readers import FileReader
            else:
                from importlib.resources.readers import FileReader

            return FileReader(types.SimpleNamespace(path=self._rewritten_names[name]))

def _write_pyc(state, co, source_stat, pyc):
    # Technically, we don't have to have the same pyc format as
    # (C)Python, since these "pycs" should never be seen by builtin
    # import. However, there's little reason deviate.
    try:
        with atomicwrites.atomic_write(pyc, mode="wb", overwrite=True, dir=tempfile.gettempdir()) as fp:
            fp.write(importlib.util.MAGIC_NUMBER)
            # as of now, bytecode header expects 32-bit numbers for size and mtime (#4903)
            mtime = int(source_stat.st_mtime) & 0xFFFFFFFF
            size = source_stat.st_size & 0xFFFFFFFF
            # "<LL" stands for 2 unsigned longs, little-ending
            fp.write(struct.pack("<LL", mtime, size))
            fp.write(marshal.dumps(co))
    except EnvironmentError as e:
        _logger.debug("error writing pyc file at {}: errno={}".format(pyc, e.errno))
        # we ignore any failure to write the cache file
        # there are many reasons, permission-denied, __pycache__ being a
        # file etc.
        return False
    return True


def _rewrite_test(fn, config):
    """read and rewrite *fn* and return the code object."""
    stat = os.stat(fn)
    with open(fn, "rb") as f:
        source = f.read()
    tree = ast.parse(source, filename=fn)
    rewrite_asserts(tree, source, fn, config)
    co = compile(tree, fn, "exec", dont_inherit=True)
    return stat, co


def _read_pyc(source, pyc, trace=lambda x: None):
    """Possibly read a pytest pyc containing rewritten code.

    Return rewritten code if successful or None if not.
    """
    try:
        fp = open(pyc, "rb")
    except IOError:
        return None
    with fp:
        try:
            stat_result = os.stat(source)
            mtime = int(stat_result.st_mtime)
            size = stat_result.st_size
            data = fp.read(12)
        except EnvironmentError as e:
            trace("_read_pyc({}): EnvironmentError {}".format(source, e))
            return None
        # Check for invalid or out of date pyc file.
        if (
            len(data) != 12
            or data[:4] != importlib.util.MAGIC_NUMBER
            or struct.unpack("<LL", data[4:]) != (mtime & 0xFFFFFFFF, size & 0xFFFFFFFF)
        ):
            trace("_read_pyc(%s): invalid or out of date pyc" % source)
            return None
        try:
            co = marshal.load(fp)
        except Exception as e:
            trace("_read_pyc({}): marshal.load error {}".format(source, e))
            return None
        if not isinstance(co, types.CodeType):
            trace("_read_pyc(%s): not a code object" % source)
            return None
        return co


def rewrite_asserts(mod, source, module_path=None, config=None):
    """Rewrite the assert statements in mod."""
    AssertionRewriter(module_path, config, source).run(mod)


def _saferepr(obj):
    """Get a safe repr of an object for assertion error messages.

    The assertion formatting (util.format_explanation()) requires
    newlines to be escaped since they are a special character for it.
    Normally assertion.util.format_explanation() does this but for a
    custom repr it is possible to contain one of the special escape
    sequences, especially '\n{' and '\n}' are likely to be present in
    JSON reprs.

    """
    return saferepr(obj).replace("\n", "\\n")

def _format_assertmsg(obj):
    """Format the custom assertion message given.

    For strings this simply replaces newlines with '\n~' so that
    util.format_explanation() will preserve them instead of escaping
    newlines.  For other objects saferepr() is used first.

    """
    # reprlib appears to have a bug which means that if a string
    # contains a newline it gets escaped, however if an object has a
    # .__repr__() which contains newlines it does not get escaped.
    # However in either case we want to preserve the newline.
    replaces = [("\n", "\n~"), ("%", "%%")]
    if not isinstance(obj, str):
        obj = saferepr(obj)
        replaces.append(("\\n", "\n~"))

    for r1, r2 in replaces:
        obj = obj.replace(r1, r2)

    return obj


def _should_repr_global_name(obj):
    if callable(obj):
        return False

    try:
        return not hasattr(obj, "__name__")
    except Exception:
        return True


def _format_boolop(explanations, is_or):
    explanation = "(" + (is_or and " or " or " and ").join(explanations) + ")"
    if isinstance(explanation, str):
        return explanation.replace("%", "%%")
    else:
        return explanation.replace(b"%", b"%%")


def _call_reprcompare(ops, results, expls, each_obj):
    # type: (Tuple[str, ...], Tuple[bool, ...], Tuple[str, ...], Tuple[object, ...]) -> str
    for i, res, expl in zip(range(len(ops)), results, expls):
        try:
            done = not res
        except Exception:
            done = True
        if done:
            break
    # pylint: disable=not-callable, undefined-loop-variable
    if util._reprcompare is not None:
        custom = util._reprcompare(ops[i], each_obj[i], each_obj[i + 1])
        if custom is not None:
            return custom
    return expl


def _call_assertion_pass(lineno, orig, expl):
    # type: (int, str, str) -> None
    # pylint: disable=not-callable
    if util._assertion_pass is not None:
        util._assertion_pass(lineno, orig, expl)


def _check_if_assertion_pass_impl():
    # type: () -> bool
    """Checks if any plugins implement the pytest_assertion_pass hook
    in order not to generate explanation unecessarily (might be expensive)"""
    return True if util._assertion_pass else False


UNARY_MAP = {ast.Not: "not %s", ast.Invert: "~%s", ast.USub: "-%s", ast.UAdd: "+%s"}

BINOP_MAP = {
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.BitAnd: "&",
    ast.LShift: "<<",
    ast.RShift: ">>",
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.FloorDiv: "//",
    ast.Mod: "%%",  # escaped for string formatting
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.Pow: "**",
    ast.Is: "is",
    ast.IsNot: "is not",
    ast.In: "in",
    ast.NotIn: "not in",
    ast.MatMult: "@",
}



def set_location(node, lineno, col_offset):
    """Set node location information recursively."""
    def _fix(node, lineno, col_offset):
        if "lineno" in node._attributes:
            node.lineno = lineno
        if "col_offset" in node._attributes:
            node.col_offset = col_offset
        for child in ast.iter_child_nodes(node):
            _fix(child, lineno, col_offset)
    _fix(node, lineno, col_offset)
    return node


def _get_assertion_exprs(src: bytes) -> Dict[int, str]:
    """Returns a mapping from {lineno: "assertion test expression"}"""
    ret = {}  # type: Dict[int, str]

    depth = 0
    lines = []  # type: List[str]
    assert_lineno = None  # type: Optional[int]
    seen_lines = set()  # type: Set[int]

    def _write_and_reset() -> None:
        nonlocal depth, lines, assert_lineno, seen_lines
        assert assert_lineno is not None
        ret[assert_lineno] = "".join(lines).rstrip().rstrip("\\")
        depth = 0
        lines = []
        assert_lineno = None
        seen_lines = set()

    tokens = tokenize.tokenize(io.BytesIO(src).readline)
    for tp, source, (lineno, offset), _, line in tokens:
        if tp == tokenize.NAME and source == "assert":
            assert_lineno = lineno
        elif assert_lineno is not None:
            # keep track of depth for the assert-message `,` lookup
            if tp == tokenize.OP and source in "([{":
                depth += 1
            elif tp == tokenize.OP and source in ")]}":
                depth -= 1

            if not lines:
                lines.append(line[offset:])
                seen_lines.add(lineno)
            # a non-nested comma separates the expression from the message
            elif depth == 0 and tp == tokenize.OP and source == ",":
                # one line assert with message
                if lineno in seen_lines and len(lines) == 1:
                    offset_in_trimmed = offset + len(lines[-1]) - len(line)
                    lines[-1] = lines[-1][:offset_in_trimmed]
                # multi-line assert with message
                elif lineno in seen_lines:
                    lines[-1] = lines[-1][:offset]
                # multi line assert with escapd newline before message
                else:
                    lines.append(line[:offset])
                _write_and_reset()
            elif tp in {tokenize.NEWLINE, tokenize.ENDMARKER}:
                _write_and_reset()
            elif lines and lineno not in seen_lines:
                lines.append(line)
                seen_lines.add(lineno)

    return ret


class AssertionRewriter(ast.NodeVisitor):
    """Assertion rewriting implementation.

    The main entrypoint is to call .run() with an ast.Module instance,
    this will then find all the assert statements and rewrite them to
    provide intermediate values and a detailed assertion error.  See
    http://pybites.blogspot.be/2011/07/behind-scenes-of-pytests-new-assertion.html
    for an overview of how this works.

    The entry point here is .run() which will iterate over all the
    statements in an ast.Module and for each ast.Assert statement it
    finds call .visit() with it.  Then .visit_Assert() takes over and
    is responsible for creating new ast statements to replace the
    original assert statement: it rewrites the test of an assertion
    to provide intermediate values and replace it with an if statement
    which raises an assertion error with a detailed explanation in
    case the expression is false and calls pytest_assertion_pass hook
    if expression is true.

    For this .visit_Assert() uses the visitor pattern to visit all the
    AST nodes of the ast.Assert.test field, each visit call returning
    an AST node and the corresponding explanation string.  During this
    state is kept in several instance attributes:

    :statements: All the AST statements which will replace the assert
       statement.

    :variables: This is populated by .variable() with each variable
       used by the statements so that they can all be set to None at
       the end of the statements.

    :variable_counter: Counter to create new unique variables needed
       by statements.  Variables are created using .variable() and
       have the form of "@py_assert0".

    :expl_stmts: The AST statements which will be executed to get
       data from the assertion.  This is the code which will construct
       the detailed assertion message that is used in the AssertionError
       or for the pytest_assertion_pass hook.

    :explanation_specifiers: A dict filled by .explanation_param()
       with %-formatting placeholders and their corresponding
       expressions to use in the building of an assertion message.
       This is used by .pop_format_context() to build a message.

    :stack: A stack of the explanation_specifiers dicts maintained by
       .push_format_context() and .pop_format_context() which allows
       to build another %-formatted string while already building one.

    This state is reset on every new assert statement visited and used
    by the other visitors.

    """

    def __init__(self, module_path, config, source):
        super().__init__()
        self.module_path = module_path
        self.config = config
        if config is not None:
            self.enable_assertion_pass_hook = config.getini(
                "enable_assertion_pass_hook"
            )
        else:
            self.enable_assertion_pass_hook = False
        self.source = source

    @functools.lru_cache(maxsize=1)
    def _assert_expr_to_lineno(self):
        return _get_assertion_exprs(self.source)

    def run(self, mod: ast.Module) -> None:
        """Find all assert statements in *mod* and rewrite them."""
        if not mod.body:
            # Nothing to do.
            return
        doc = getattr(mod, "docstring", None)
        expect_docstring = doc is None
        if doc is not None and self.is_rewrite_disabled(doc):
            return
        pos = 0
        lineno = 1
        for item in mod.body:
            if (
                expect_docstring
                and isinstance(item, ast.Expr)
                and isinstance(item.value, ast.Str)
            ):
                doc = item.value.s
                if self.is_rewrite_disabled(doc):
                    return
                expect_docstring = False
            elif (
                not isinstance(item, ast.ImportFrom)
                or item.level > 0
                or item.module != "__future__"
            ):
                lineno = item.lineno
                break
            pos += 1
        else:
            lineno = item.lineno
        # Insert some special imports at the top of the module but after any
        # docstrings and __future__ imports.
        if sys.version_info < (3, 10):
            aliases = [
                ast.alias("builtins", "@py_builtins"),
                ast.alias("dessert.rewrite", "@dessert_ar"),
            ]
        else:
            aliases = [
                ast.alias("builtins", "@py_builtins", lineno=lineno, col_offset=0),
                ast.alias("dessert.rewrite", "@dessert_ar", lineno=lineno, col_offset=0),
            ]
        imports = [
            ast.Import([alias], lineno=lineno, col_offset=0) for alias in aliases
        ]
        mod.body[pos:pos] = imports
        # Collect asserts.
        nodes = [mod]  # type: List[ast.AST]
        while nodes:
            node = nodes.pop()
            for name, field in ast.iter_fields(node):
                if isinstance(field, list):
                    new = []  # type: List
                    for _, child in enumerate(field):
                        if isinstance(child, ast.Assert):
                            # Transform assert.
                            new.extend(self.visit(child))
                        else:
                            new.append(child)
                            if isinstance(child, ast.AST):
                                nodes.append(child)
                    setattr(node, name, new)
                elif (
                    isinstance(field, ast.AST)
                    # Don't recurse into expressions as they can't contain
                    # asserts.
                    and not isinstance(field, ast.expr)
                ):
                    nodes.append(field)

    @staticmethod
    def is_rewrite_disabled(docstring):
        return "PYTEST_DONT_REWRITE" in docstring

    def variable(self):
        """Get a new variable."""
        # Use a character invalid in python identifiers to avoid clashing.
        name = "@py_assert" + str(next(self.variable_counter))
        self.variables.append(name)
        return name

    def assign(self, expr):
        """Give *expr* a name."""
        name = self.variable()
        self.statements.append(ast.Assign([ast.Name(name, ast.Store())], expr))
        return ast.Name(name, ast.Load())

    def display(self, expr):
        """Call saferepr on the expression."""
        return self.helper("_saferepr", expr)

    def helper(self, name, *args):
        """Call a helper in this module."""
        py_name = ast.Name("@dessert_ar", ast.Load())
        attr = ast.Attribute(py_name, name, ast.Load())
        return ast.Call(attr, list(args), [])

    def builtin(self, name):
        """Return the builtin called *name*."""
        builtin_name = ast.Name("@py_builtins", ast.Load())
        return ast.Attribute(builtin_name, name, ast.Load())

    def explanation_param(self, expr):
        """Return a new named %-formatting placeholder for expr.

        This creates a %-formatting placeholder for expr in the
        current formatting context, e.g. ``%(py0)s``.  The placeholder
        and expr are placed in the current format context so that it
        can be used on the next call to .pop_format_context().

        """
        specifier = "py" + str(next(self.variable_counter))
        self.explanation_specifiers[specifier] = expr
        return "%(" + specifier + ")s"

    def push_format_context(self):
        """Create a new formatting context.

        The format context is used for when an explanation wants to
        have a variable value formatted in the assertion message.  In
        this case the value required can be added using
        .explanation_param().  Finally .pop_format_context() is used
        to format a string of %-formatted values as added by
        .explanation_param().

        """
        self.explanation_specifiers = {}  # type: Dict[str, ast.expr]
        self.stack.append(self.explanation_specifiers)

    def pop_format_context(self, expl_expr):
        """Format the %-formatted string with current format context.

        The expl_expr should be an ast.Str instance constructed from
        the %-placeholders created by .explanation_param().  This will
        add the required code to format said string to .expl_stmts and
        return the ast.Name instance of the formatted string.

        """
        current = self.stack.pop()
        if self.stack:
            self.explanation_specifiers = self.stack[-1]
        keys = [ast.Str(key) for key in current.keys()]
        format_dict = ast.Dict(keys, list(current.values()))
        form = ast.BinOp(expl_expr, ast.Mod(), format_dict)
        name = "@py_format" + str(next(self.variable_counter))
        if self.enable_assertion_pass_hook:
            self.format_variables.append(name)
        self.expl_stmts.append(ast.Assign([ast.Name(name, ast.Store())], form))
        return ast.Name(name, ast.Load())

    def generic_visit(self, node):
        """Handle expressions we don't have custom code for."""
        assert isinstance(node, ast.expr)
        res = self.assign(node)
        return res, self.explanation_param(self.display(res))

    def visit_Assert(self, assert_):
        """Return the AST statements to replace the ast.Assert instance.

        This rewrites the test of an assertion to provide
        intermediate values and replace it with an if statement which
        raises an assertion error with a detailed explanation in case
        the expression is false.

        """
        if isinstance(assert_.test, ast.Tuple) and len(assert_.test.elts) >= 1:
            from dessert.warning_types import DessertAssertRewriteWarning
            import warnings

            # Ignore type: typeshed bug https://github.com/python/typeshed/pull/3121
            warnings.warn_explicit(  # type: ignore
                DessertAssertRewriteWarning(
                    "assertion is always true, perhaps remove parentheses?"
                ),
                category=None,
                filename=self.module_path,
                lineno=assert_.lineno,
            )

        self.statements = []  # type: List[ast.stmt]
        self.variables = []  # type: List[str]
        self.variable_counter = itertools.count()

        if self.enable_assertion_pass_hook:
            self.format_variables = []  # type: List[str]

        self.stack = []  # type: List[Dict[str, ast.expr]]
        self.expl_stmts = []  # type: List[ast.stmt]
        self.push_format_context()
        # Rewrite assert into a bunch of statements.
        top_condition, explanation = self.visit(assert_.test)
        # If in a test module, check if directly asserting None, in order to warn [Issue #3191]
        if self.module_path is not None:
            self.statements.append(
                self.warn_about_none_ast(
                    top_condition, module_path=self.module_path, lineno=assert_.lineno
                )
            )

        if self.enable_assertion_pass_hook:  # Experimental pytest_assertion_pass hook
            negation = ast.UnaryOp(ast.Not(), top_condition)
            msg = self.pop_format_context(ast.Str(explanation))

            # Failed
            if assert_.msg:
                assertmsg = self.helper("_format_assertmsg", assert_.msg)
                gluestr = "\n>assert "
            else:
                assertmsg = ast.Str("")
                gluestr = "assert "
            err_explanation = ast.BinOp(ast.Str(gluestr), ast.Add(), msg)
            err_msg = ast.BinOp(assertmsg, ast.Add(), err_explanation)
            err_name = ast.Name("AssertionError", ast.Load())
            fmt = self.helper("_format_explanation", err_msg)
            exc = ast.Call(err_name, [fmt], [])
            raise_ = ast.Raise(exc, None)
            statements_fail = []
            statements_fail.extend(self.expl_stmts)
            statements_fail.append(raise_)

            # Passed
            fmt_pass = self.helper("_format_explanation", msg)
            orig = self._assert_expr_to_lineno()[assert_.lineno]
            hook_call_pass = ast.Expr(
                self.helper(
                    "_call_assertion_pass",
                    ast.Num(assert_.lineno),
                    ast.Str(orig),
                    fmt_pass,
                )
            )
            # If any hooks implement assert_pass hook
            hook_impl_test = ast.If(
                self.helper("_check_if_assertion_pass_impl"),
                self.expl_stmts + [hook_call_pass],
                [],
            )
            statements_pass = [hook_impl_test]

            # Test for assertion condition
            main_test = ast.If(negation, statements_fail, statements_pass)
            self.statements.append(main_test)
            if self.format_variables:
                variables = [
                    ast.Name(name, ast.Store()) for name in self.format_variables
                ]
                clear_format = ast.Assign(variables, ast.NameConstant(None))
                self.statements.append(clear_format)

        else:  # Original assertion rewriting
            # Create failure message.
            body = self.expl_stmts
            negation = ast.UnaryOp(ast.Not(), top_condition)
            self.statements.append(ast.If(negation, body, []))
            if assert_.msg:
                assertmsg = self.helper("_format_assertmsg", assert_.msg)
                explanation = "\n>assert " + explanation
            else:
                assertmsg = ast.Str("")
                explanation = "assert " + explanation

            if _MARK_ASSERTION_INTROSPECTION:
                explanation = "dessert* " + explanation

            template = ast.BinOp(assertmsg, ast.Add(), ast.Str(explanation))
            msg = self.pop_format_context(template)
            fmt = self.helper("_format_explanation", msg, assertmsg)
            err_name = ast.Name("AssertionError", ast.Load())
            exc = ast.Call(err_name, [fmt], [])
            raise_ = ast.Raise(exc, None)

            body.append(raise_)

        # Clear temporary variables by setting them to None.
        if self.variables:
            variables = [ast.Name(name, ast.Store()) for name in self.variables]
            clear = ast.Assign(variables, ast.NameConstant(None))
            self.statements.append(clear)
        # Fix line numbers.
        for stmt in self.statements:
            set_location(stmt, assert_.lineno, assert_.col_offset)
        return self.statements

    def warn_about_none_ast(self, node, module_path, lineno):
        """
        Returns an AST issuing a warning if the value of node is `None`.
        This is used to warn the user when asserting a function that asserts
        internally already.
        See issue #3191 for more details.
        """
        val_is_none = ast.Compare(node, [ast.Is()], [ast.NameConstant(None)])

        exc_call = ast.Call(
            func=ast.Name("DessertAssertRewriteWarning", ast.Load()),
            args=[ast.Str('asserting the value None, please use "assert is None"')],
            keywords=[]
        )
        warn_call = ast.Call(
            func=ast.Name("warn_explicit", ast.Load()),
            args=[exc_call, ast.NameConstant(None), ast.Str(module_path), ast.Num(lineno)],
            keywords=[]
        )
        body = [
            ast.ImportFrom("dessert.warning_types", [ast.alias("DessertAssertRewriteWarning")], level=0),
            ast.ImportFrom("warnings", [ast.alias("warn_explicit")], level=0),
            ast.Expr(warn_call),
        ]

        return ast.If(val_is_none, body, [])

    def visit_Name(self, name):
        # Display the repr of the name if it's a local variable or
        # _should_repr_global_name() thinks it's acceptable.
        locs = ast.Call(self.builtin("locals"), [], [])
        inlocs = ast.Compare(ast.Str(name.id), [ast.In()], [locs])
        dorepr = self.helper("_should_repr_global_name", name)
        test = ast.BoolOp(ast.Or(), [inlocs, dorepr])
        expr = ast.IfExp(test, self.display(name), ast.Str(name.id))
        return name, self.explanation_param(expr)

    def visit_BoolOp(self, boolop):
        res_var = self.variable()
        expl_list = self.assign(ast.List([], ast.Load()))
        app = ast.Attribute(expl_list, "append", ast.Load())
        is_or = int(isinstance(boolop.op, ast.Or))
        body = save = self.statements
        fail_save = self.expl_stmts
        levels = len(boolop.values) - 1
        self.push_format_context()
        # Process each operand, short-circuiting if needed.
        for i, v in enumerate(boolop.values):
            if i:
                fail_inner = []  # type: List[ast.stmt]
                # cond is set in a prior loop iteration below
                self.expl_stmts.append(ast.If(cond, fail_inner, []))  # pylint: disable=possibly-used-before-assignment
                self.expl_stmts = fail_inner
            self.push_format_context()
            res, expl = self.visit(v)
            body.append(ast.Assign([ast.Name(res_var, ast.Store())], res))
            expl_format = self.pop_format_context(ast.Str(expl))
            call = ast.Call(app, [expl_format], [])
            self.expl_stmts.append(ast.Expr(call))
            if i < levels:
                cond = res  # type: ast.expr
                if is_or:
                    cond = ast.UnaryOp(ast.Not(), cond)
                inner = []  # type: List[ast.stmt]
                self.statements.append(ast.If(cond, inner, []))
                self.statements = body = inner
        self.statements = save
        self.expl_stmts = fail_save
        expl_template = self.helper("_format_boolop", expl_list, ast.Num(is_or))
        expl = self.pop_format_context(expl_template)
        return ast.Name(res_var, ast.Load()), self.explanation_param(expl)

    def visit_UnaryOp(self, unary):
        pattern = UNARY_MAP[unary.op.__class__]
        operand_res, operand_expl = self.visit(unary.operand)
        res = self.assign(ast.UnaryOp(unary.op, operand_res))
        return res, pattern % (operand_expl,)

    def visit_BinOp(self, binop):
        symbol = BINOP_MAP[binop.op.__class__]
        left_expr, left_expl = self.visit(binop.left)
        right_expr, right_expl = self.visit(binop.right)
        explanation = "({} {} {})".format(left_expl, symbol, right_expl)
        res = self.assign(ast.BinOp(left_expr, binop.op, right_expr))
        return res, explanation

    def visit_Call(self, call):
        """
        visit `ast.Call` nodes
        """
        new_func, func_expl = self.visit(call.func)
        arg_expls = []
        new_args = []
        new_kwargs = []
        for arg in call.args:
            res, expl = self.visit(arg)
            arg_expls.append(expl)
            new_args.append(res)
        for keyword in call.keywords:
            res, expl = self.visit(keyword.value)
            new_kwargs.append(ast.keyword(keyword.arg, res))
            if keyword.arg:
                arg_expls.append(keyword.arg + "=" + expl)
            else:  # **args have `arg` keywords with an .arg of None
                arg_expls.append("**" + expl)

        expl = "{}({})".format(func_expl, ", ".join(arg_expls))
        new_call = ast.Call(new_func, new_args, new_kwargs)
        res = self.assign(new_call)
        res_expl = self.explanation_param(self.display(res))
        outer_expl = "{}\n{{{} = {}\n}}".format(res_expl, res_expl, expl)
        return res, outer_expl

    def visit_Starred(self, starred):
        # From Python 3.5, a Starred node can appear in a function call
        res, expl = self.visit(starred.value)
        new_starred = ast.Starred(res, starred.ctx)
        return new_starred, "*" + expl

    def visit_Attribute(self, attr):
        if not isinstance(attr.ctx, ast.Load):
            return self.generic_visit(attr)
        value, value_expl = self.visit(attr.value)
        res = self.assign(ast.Attribute(value, attr.attr, ast.Load()))
        res_expl = self.explanation_param(self.display(res))
        pat = "%s\n{%s = %s.%s\n}"
        expl = pat % (res_expl, res_expl, value_expl, attr.attr)
        return res, expl

    def visit_Compare(self, comp: ast.Compare):
        self.push_format_context()
        left_res, left_expl = self.visit(comp.left)
        if isinstance(comp.left, (ast.Compare, ast.BoolOp)):
            left_expl = "({})".format(left_expl)
        res_variables = [self.variable() for i in range(len(comp.ops))]
        load_names = [ast.Name(v, ast.Load()) for v in res_variables]
        store_names = [ast.Name(v, ast.Store()) for v in res_variables]
        it = zip(range(len(comp.ops)), comp.ops, comp.comparators)
        expls = []
        syms = []
        results = [left_res]
        for i, op, next_operand in it:
            next_res, next_expl = self.visit(next_operand)
            if isinstance(next_operand, (ast.Compare, ast.BoolOp)):
                next_expl = "({})".format(next_expl)
            results.append(next_res)
            sym = BINOP_MAP[op.__class__]
            syms.append(ast.Str(sym))
            expl = "{} {} {}".format(left_expl, sym, next_expl)
            expls.append(ast.Str(expl))
            res_expr = ast.Compare(left_res, [op], [next_res])
            self.statements.append(ast.Assign([store_names[i]], res_expr))
            left_res, left_expl = next_res, next_expl
        # Use pytest.assertion.util._reprcompare if that's available.
        expl_call = self.helper(
            "_call_reprcompare",
            ast.Tuple(syms, ast.Load()),
            ast.Tuple(load_names, ast.Load()),
            ast.Tuple(expls, ast.Load()),
            ast.Tuple(results, ast.Load()),
        )
        if len(comp.ops) > 1:
            res = ast.BoolOp(ast.And(), load_names)  # type: ast.expr
        else:
            res = load_names[0]
        return res, self.explanation_param(self.pop_format_context(expl_call))


def try_mkdir(cache_dir):
    """Attempts to create the given directory, returns True if successful"""
    try:
        os.mkdir(cache_dir)
    except FileExistsError:
        # Either the __pycache__ directory already exists (the
        # common case) or it's blocked by a non-dir node. In the
        # latter case, we'll ignore it in _write_pyc.
        return True
    except (FileNotFoundError, NotADirectoryError):
        # One of the path components was not a directory, likely
        # because we're in a zip file.
        return False
    except PermissionError:
        return False
    except OSError as e:
        # as of now, EROFS doesn't have an equivalent OSError-subclass
        if e.errno == errno.EROFS:
            return False
        raise
    return True
