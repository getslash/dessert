
![Build Status](https://github.com/vmalloc/dessert/actions/workflows/test.yml/badge.svg?branch=develop)
![Version](https://img.shields.io/pypi/v/dessert.svg)

Overview
========

Dessert is a utility library enabling Python code to introspect assertions raised via the `assert` statement.

It is a standalone version of the introspection code from [pytest](http://pytest.org ), and all credit is due to Holger Krekel and the pytest developers for this code.

Usage
=====

Using dessert is fairly simple:

```python

>>> with open(tmp_filename, "w") as f:
...     _ = f.write("""
... def func():
...     def x():
...         return 1
...     def y():
...         return -1
...     assert y() == x()
... """)

>>> import emport
>>> import dessert
>>> with dessert.rewrite_assertions_context():
...     module = emport.import_file(tmp_filename)
>>> module.func() # doctest: +IGNORE_EXCEPTION_DETAIL +ELLIPSIS
Traceback (most recent call last):
    ...
    assert y() == x()
AssertionError: assert -1 == 1
+  where -1 = <function y at ...>()
+  and   1 = <function x at ...>()
```

Licence
=======

Dessert is released under the MIT license. It is 99% based on pytest, which is released under the MIT license.
