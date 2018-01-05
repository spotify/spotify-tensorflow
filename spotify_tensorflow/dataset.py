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

import json
import os

import tensorflow as tf
from tensorflow.python.lib.io import file_io

FLAGS = tf.flags.FLAGS


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
    def _get_featran_example_dataset(dir_path,
                                     feature_desc_path=None,
                                     feature_mapping_fn=None,
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
        input_files = [pjoin(dir_path, x) for x in filter(lambda f: not f.startswith("_"), flist)]
        feature_desc_path = feature_desc_path or Datasets.__get_default_feature_desc_path(dir_path)
        filenames = tf.placeholder_with_default(input_files, shape=[None])
        feature_mapping_fn = feature_mapping_fn or Datasets.__get_default_feature_mapping_fn()
        num_features, parse_fn = Datasets.get_parse_proto_function(feature_desc_path,
                                                                   feature_mapping_fn)
        dataset = tf.data.Dataset.from_tensor_slices(filenames)

        tf_config = os.environ.get("TF_CONFIG")

        # If TF_CONFIG is not available don't bother sharding
        if tf_config is not None:
            tf_config_json = json.loads(tf_config)
            tf.logging.info("Found TF_CONFIG: %s" % tf_config)
            num_workers = len(tf_config_json.get("cluster", {}).get("worker", []))
            worker_index = tf_config_json.get("task", {}).get("index", None)
            if worker_index is not None:
                tf.logging.info("Sharding dataset on worker_index=%s out of %s workers"
                                % (worker_index, num_workers))
                dataset = dataset.shard(num_workers, worker_index)

        if FLAGS.interleaving_threads > 0:
            dataset = dataset.interleave(lambda f: tf.data.TFRecordDataset(f, compression_type),
                                         cycle_length=FLAGS.interleaving_threads,
                                         block_length=FLAGS.interleaving_block_length)
        else:
            dataset = tf.data.TFRecordDataset(filenames, compression_type)

        dataset = dataset.map(parse_fn, num_parallel_calls=FLAGS.parsing_threads)
        return dataset, DatasetContext(filenames_placeholder=filenames, num_features=num_features)

    @staticmethod
    def __get_default_feature_mapping_fn():
        return lambda l: tf.FixedLenFeature((), tf.int64, default_value=0)

    @staticmethod
    def __get_default_feature_desc_path(dir_path):
        from os.path import join as pjoin
        return pjoin(dir_path, Datasets._default_feature_desc_filename)

    @classmethod
    def mk_training_iter(cls, training_data_dir, feature_mapping_fn):
        """Make a training `Dataset` iterator.

        Args:
            training_data_dir: a directory contains training data.
            feature_mapping_fn: A function which maps `FeatureInfo` to `FixedLenFeature` or
                `VarLenFeature` values. Default maps all features to
                tf.FixedLenFeature((), feature_info.type, default_value=0).

        Returns:
            A `Dataset` that should be used for training purposes.
        """
        with tf.name_scope("training-input"):
            train_dataset, _ = cls._get_featran_example_dataset(
                training_data_dir,
                feature_mapping_fn=feature_mapping_fn)
            return cls._mk_iterator(train_dataset)

    @classmethod
    def mk_eval_iter(cls, eval_data_dir, feature_mapping_fn):
        """Make an evaluation `Dataset` iterator.

        Args:
            eval_data_dir: a directory contains evaluation data.
            feature_mapping_fn: A function which maps `FeatureInfo` to `FixedLenFeature` or
                `VarLenFeature` values. Default maps all features to
                tf.FixedLenFeature((), feature_info.type, default_value=0).

        Returns:
            A `Dataset` that should be used for evaluation purposes.
        """
        with tf.name_scope("evaluation-input"):
            eval_dataset, _ = cls._get_featran_example_dataset(
                eval_data_dir,
                feature_mapping_fn=feature_mapping_fn)
            return cls._mk_iterator(eval_dataset)

    @staticmethod
    def _mk_iterator(dataset):
        if FLAGS.shuffle_buffer_size > 0:
            dataset = dataset.shuffle(FLAGS.shuffle_buffer_size)

        if FLAGS.batch_size > 0:
            dataset = dataset.batch(FLAGS.batch_size)

        dataset = dataset.take(FLAGS.take_count)

        if FLAGS.prefetch_buffer_size > 0:
            dataset = dataset.prefetch(FLAGS.prefetch_buffer_size)
        # TODO(rav): evaluate the use of initializable iterator for more epochs?
        iterator = dataset.make_one_shot_iterator()
        return iterator
