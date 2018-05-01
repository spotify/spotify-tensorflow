# spotify-tensorflow

[![Build Status](https://img.shields.io/circleci/project/github/spotify/spotify-tensorflow/master.svg)](https://circleci.com/gh/spotify/spotify-tensorflow)
[![codecov.io](https://codecov.io/github/spotify/spotify-tensorflow/coverage.svg?branch=master)](https://codecov.io/github/spotify/spotify-tensorflow?branch=master)
[![GitHub license](https://img.shields.io/github/license/spotify/spotify-tensorflow.svg)](./LICENSE)
[![PyPI version](https://badge.fury.io/py/spotify_tensorflow.svg)](https://badge.fury.io/py/spotify_tensorflow)

## Raison d'être:

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
 * TensorFlow training via tf.Example dataset
