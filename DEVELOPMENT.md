# Dessert Development Process

Dessert follows the *pytest* project, and isolates the assertion rewriting mechanism introduced in pytest for standalone projects.

## Fetching Up-to-date Changes

First, check out the `pytest` branch:

```
$ git checkout pytest
```

Then, pull that branch to receive the latest changes:

```
$ git pull git@github.com:pytest-dev/pytest.git master
```

Merge the changes into a temporary update branch:

```
$ git checkout -b update
$ git merge pytest
```

You will have to resolve merge conflicts:

```
$ git rm -rf testing extra doc AUTHORS CHANGELOG.rst CONTRIBUTING.rst HOWTORELEASE.rst README.rst README.rst appveyor.yml setup.cfg _pytest .github
$ git reset .travis.yml .gitignore MANIFEST.in setup.py tox.ini 
$ git checkout --ours .travis.yml .gitignore MANIFEST.in setup.py tox.ini 
```

Then the remaining file(s) should be mostly dessert/rewrite.py and dessert/util.py. Those are the ones with actual logica that needs to be resolved. Resolve them and commit the merge.
