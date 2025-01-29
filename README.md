# YMDF model

YMDF stands for **Y**oung **M** **D**warf **F**lare.

<!-- MarkdownTOC -->

- [Credits](#credits)
- [Installing](#installing)
    - [From PyPI](#from-pypi)
    - [From sources](#from-sources)
        - [Building a wheel](#building-a-wheel)
- [Tests](#tests)

<!-- /MarkdownTOC -->

## Credits

The package is based on [AltaiPony](https://github.com/ekaterinailin/AltaiPony) by [Ekaterina Ilin](https://ekaterinailin.github.io/).

## Installing

### From PyPI

``` sh
$ pip install ymdf
```

### From sources

``` sh
$ cd /path/to/ymdf/
$ pip install .
```

Add an `-e` argument, if you'd like to automatically update your locally installed package by pulling from the repository or/and if you intend to modify the sources:

``` sh
$ pip install -e .
```

#### Building a wheel

You can also build a wheel and distribute/install that instead:

``` sh
$ cd /path/to/ymdf/
$ python -m build
$ pip install ./dist/ymdf-*.whl
```

## Tests

To run tests:

``` sh
$ pip install pytest

$ python -m pytest ./src/phab/tests/*[^_*].py
$ python -m pytest ./src/phab/tests/model.py
$ python -m pytest ./src/phab/tests/some.py -k "test_some_thing"
```
