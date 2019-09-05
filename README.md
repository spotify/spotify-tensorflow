# spotify-tensorflow

[![Build Status](https://img.shields.io/circleci/project/github/spotify/spotify-tensorflow/master.svg)](https://circleci.com/gh/spotify/spotify-tensorflow)
[![Coverage](https://img.shields.io/codecov/c/github/spotify/spotify-tensorflow/master.svg?style=flat)](https://codecov.io/github/spotify/spotify-tensorflow?branch=master)
[![GitHub license](https://img.shields.io/github/license/spotify/spotify-tensorflow.svg)](./LICENSE)
[![PyPI version](https://badge.fury.io/py/spotify_tensorflow.svg)](https://badge.fury.io/py/spotify_tensorflow)

## Purpose:

Provide Spotify specific TensorFlow helpers.

## Features

 * tf.data integration with [Featran](https://github.com/spotify/featran)
 * common Dataset API to read:
   * TFRecord datasets as tf.Tensor
   * TFRecord datasets as Pandas DataFrame
   * TFRecord datasets as python dict
 * tf.Graph freezing utilities
 * TensorFlow integration with [Luigi](https://github.com/spotify/luigi)

## Examples:

Check examples in the [examples directory](https://github.com/spotify/spotify-tensorflow/tree/master/examples),
currently they include:

 * sklearn training via Pandas DataFrame dataset
 * XGBoost training via tf.Example dataset
 * TensorFlow training via tf.Example dataset

To run the examples:

```sh
$ pip install -e .[examples]
$ bin/run-examples
```

## Development:

This project uses `tox`.

```sh
$ pip install tox
```

To see all `testenv`'s:

```sh
$ tox -l
mypy
lint
examples-py27
examples-py35
test-py27
test-py35
upload-coverage
license
```

To run the tests:

```
tox -e test
```

To release:

```
git commit --allow-empty -m "Release x.y.z"
git tag x.y.z
git push --tags  origin master
```

Then upload to pypi:

```
python setup.py sdist upload -r pypi
```
