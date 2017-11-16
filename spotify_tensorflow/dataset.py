# -*- coding: utf-8 -*-
#
# Copyright 2017 Spotify AB.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

from __future__ import absolute_import, division, print_function

import multiprocessing as mp

import tensorflow as tf
from tensorflow.python.lib.io import file_io


class DatasetContext(object):
    """Holds additional information about/from Dataset parsing.

    Attributes:
        filenames_placeholder: A placeholder for Dataset file inputs.
        num_features: Number of features available in the Dataset.
    """

    def __init__(self, filenames_placeholder, num_features):
        self.filenames_placeholder = filenames_placeholder
        self.num_features = num_features


class Datasets(object):
    _default_feature_desc_filename = "_feature_desc"

    @staticmethod
    def get_parse_proto_function(feature_desc_path, feature_mapping_fn):
        """Get a function to parse `Example` proto using given features specifications.

        Args:
            feature_desc_path: filepath to a feature description file.
            feature_mapping_fn: A function which maps feature spec line to `FixedLenFeature` or
                `VarLenFeature` values.

        Returns:
            A Tuple of two elements: (number of features, parse function). Parse function takes a
            single element - a scalar string Tensor, a single serialized Example.
        """
        assert isinstance(feature_desc_path, str), \
            "dir_path is not a String: %r" % feature_desc_path
        assert file_io.file_exists(feature_desc_path), \
            "feature desc `%s` does not exist" % feature_desc_path

        def get_features(fpath):
            features = {}
            with file_io.FileIO(fpath, "r") as f:
                for feature_spec_line in f.readlines():
                    feature_spec_line = feature_spec_line.strip()
                    features[feature_spec_line] = feature_mapping_fn(feature_spec_line)
            return features

        feature_spec = get_features(feature_desc_path)

        def _parse_function(example_proto):
            return tf.parse_single_example(example_proto, feature_spec)

        return len(feature_spec), _parse_function

    @staticmethod
    def get_featran_example_dataset(dir_path,
                                    feature_desc_path=None,
                                    feature_mapping_fn=None,
                                    num_threads=mp.cpu_count(),
                                    num_threads_per_file=mp.cpu_count(),
                                    block_length=32,
                                    compression_type="ZLIB"):
        """Get `Dataset` of parsed `Example` protos.

        Args:
            dir_path: Directory path containing features.
            feature_desc_path: Filepath to feature description file. Default is `_feature_desc`
                inside `dir_path`.
            feature_mapping_fn: A function which maps feature spec line to `FixedLenFeature` or
                `VarLenFeature` values. Default maps all features to
                tf.FixedLenFeature((), tf.int64, default_value=0).
            compression_type: A `tf.string` scalar evaluating to one of `""` (no compression)
                `"ZLIB"`, or `"GZIP"`.
            num_threads: A `tf.int32` scalar or `tf.Tensor`, represents number of files to process
                concurrently.
            num_threads_per_file: A `tf.int32` scalar or `tf.Tensor`, represents number of threads
                used concurrently per file.
            block_length: A `tf.int32` scalar or `tf.Tensor`, represents buffer size for results
                from any of the parsing threads.

        Returns:
            A Tuple of two elements: (dataset, dataset_context). First element is a `Dataset`, which
            holds results of the parsing of `Example` protos. Second element holds a
            `DatasetContext` (see doc of `DatasetContext`).
        """
        assert isinstance(dir_path, str), "dir_path is not a String: %r" % dir_path
        assert isinstance(feature_desc_path, str) or feature_desc_path is None, \
            "dir_path is not a String: %r" % feature_desc_path
        assert file_io.file_exists(dir_path), "directory `%s` does not exist" % dir_path
        assert file_io.is_directory(dir_path), "`%s` is not a directory" % dir_path
        if feature_desc_path:
            assert file_io.file_exists(feature_desc_path), \
                "feature desc `%s` does not exist" % feature_desc_path

        from os.path import join as pjoin
        flist = file_io.list_directory(dir_path)
        input_files = [pjoin(dir_path, x) for x in filter(lambda x: not x.startswith("_"), flist)]
        if feature_desc_path is None:
            feature_desc_path = pjoin(dir_path, Datasets._default_feature_desc_filename)
        filenames = tf.placeholder_with_default(input_files, shape=[None])
        feature_mapping_fn = feature_mapping_fn or Datasets.__get_default_feature_mapping_fn()
        num_features, parse_fn = Datasets.get_parse_proto_function(feature_desc_path,
                                                                   feature_mapping_fn)
        dataset = (tf.data.Dataset.from_tensor_slices(filenames)
                   .interleave(lambda x: tf.data.TFRecordDataset(x, compression_type)
                               .map(parse_fn, num_parallel_calls=num_threads_per_file),
                               cycle_length=num_threads, block_length=block_length))
        return dataset, DatasetContext(filenames_placeholder=filenames, num_features=num_features)

    @staticmethod
    def __get_default_feature_mapping_fn():
        return lambda l: tf.FixedLenFeature((), tf.int64, default_value=0)

    @staticmethod
    def mk_dataset_training(training_data_dir, feature_mapping_fn):
        """Make a training `Dataset`.

        Args:
            training_data_dir: a directory contains training data.

        Returns:
            A `Dataset` that should be used for training purposes.
        """
        with tf.name_scope("training-input"):
            train_dataset, _ = Datasets.get_featran_example_dataset(
                training_data_dir,
                feature_mapping_fn=feature_mapping_fn)
            return train_dataset

    @staticmethod
    def mk_dataset_eval(eval_data_dir, feature_mapping_fn):
        """Make an evaluation `Dataset`.

        Args:
            eval_data_dir: a directory contains evaluation data.

        Returns:
            A `Dataset` that should be used for evaluation purposes.
        """
        with tf.name_scope("evaluation-input"):
            eval_dataset, _ = Datasets.get_featran_example_dataset(
                eval_data_dir,
                feature_mapping_fn=feature_mapping_fn)
            return eval_dataset
