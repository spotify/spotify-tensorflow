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
from collections import namedtuple
from os.path import join as pjoin

import tensorflow as tf
from tensorflow.python.lib.io import file_io

FLAGS = tf.flags.FLAGS


class DatasetContext(namedtuple("DatasetContext", ["filenames", "features"])):
    """Holds additional information about/from Dataset parsing.

    Attributes:
        filenames: A placeholder for Dataset file inputs.
        features: Map features and their type available in the Dataset.
    """


class Datasets(object):

    @staticmethod
    def _parse_feature_desc(feature_desc_path, dir_path):
        """Extract the names of the features out of the feature description file.

        Args:
            feature_desc_path: filepath to a feature description file.
            dir_path: Directory path containing features.

        Returns:
            A list containing the names of the features.
        """
        if feature_desc_path is None:
            feature_desc_path = pjoin(dir_path, "_feature_desc")
        else:
            assert isinstance(feature_desc_path, str), \
                "dir_path is not a String: %r" % feature_desc_path
            assert file_io.file_exists(feature_desc_path), \
                "feature desc `%s` does not exist" % feature_desc_path

        with file_io.FileIO(feature_desc_path, "r") as f:
            return [feature_name.strip() for feature_name in f.readlines()]

    @classmethod
    def _get_featran_example_dataset(cls,
                                     dir_path,
                                     feature_desc_path=None,
                                     feature_mapping_fn=None,
                                     compression_type="ZLIB"):
        """Get `Dataset` of parsed `Example` protos.

        Args:
            dir_path: Directory path containing features.
            feature_desc_path: Filepath to feature description file. Default is `_feature_desc`
                inside `dir_path`.
            feature_mapping_fn: A function which maps feature names to `FixedLenFeature` or
                `VarLenFeature` values. Default maps all features to
                tf.FixedLenFeature((), tf.int64, default_value=0).
            compression_type: A `tf.string` scalar evaluating to one of `""` (no compression)
                `"ZLIB"`, or `"GZIP"`.

        Returns:
            A Tuple of two elements: (dataset, dataset_context). First element is a `Dataset`, which
            holds results of the parsing of `Example` protos. Second element holds a
            `DatasetContext` (see doc of `DatasetContext`).
        """
        filenames = cls._get_tfrecord_filenames(dir_path)
        feature_names = cls._parse_feature_desc(feature_desc_path, dir_path)

        if FLAGS.interleaving_threads > 0:
            dataset = tf.data.Dataset.from_tensor_slices(filenames)
            dataset = dataset.interleave(lambda f: tf.data.TFRecordDataset(f, compression_type),
                                         cycle_length=FLAGS.interleaving_threads,
                                         block_length=FLAGS.interleaving_block_length)
        else:
            dataset = tf.data.TFRecordDataset(filenames, compression_type)

        dataset = cls._try_enable_sharding(dataset)

        feature_mapping_fn = feature_mapping_fn or cls._get_default_feature_mapping_fn
        features = feature_mapping_fn(feature_names)

        def _parse_function(example_proto):
            return tf.parse_single_example(example_proto, features)

        dataset = dataset.map(_parse_function, num_parallel_calls=FLAGS.parsing_threads)
        return dataset, DatasetContext(filenames, features)

    @staticmethod
    def _get_tfrecord_filenames(dir_path):
        assert isinstance(dir_path, str), "dir_path is not a String: %r" % dir_path
        assert file_io.file_exists(dir_path), "directory `%s` does not exist" % dir_path
        assert file_io.is_directory(dir_path), "`%s` is not a directory" % dir_path
        flist = file_io.list_directory(dir_path)
        input_files = [pjoin(dir_path, x) for x in filter(lambda f: not f.startswith("_"), flist)]
        filenames = tf.placeholder_with_default(input_files, shape=[None])
        return filenames

    @staticmethod
    def _get_default_feature_mapping_fn(feature_names):
        fm = [(name, tf.FixedLenFeature((), tf.int64, default_value=0)) for name in feature_names]
        return dict(fm)

    @staticmethod
    def _try_enable_sharding(dataset):
        tf_config = os.environ.get("TF_CONFIG")

        if tf_config is not None:
            tf_config_json = json.loads(tf_config)
            tf.logging.info("Found TF_CONFIG: %s" % tf_config)
            num_workers = len(tf_config_json.get("cluster", {}).get("worker", []))
            worker_index = tf_config_json.get("task", {}).get("index", None)
            if worker_index is not None:
                tf.logging.info("Sharding dataset on worker_index=%s out of %s workers"
                                % (worker_index, num_workers))
                return dataset.shard(num_workers, worker_index)
        return dataset

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
