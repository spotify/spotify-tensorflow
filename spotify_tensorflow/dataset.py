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
import multiprocessing as mp
import os
from collections import namedtuple

import tensorflow as tf
from tensorflow.python.lib.io import file_io

from .tf_record_spec_parser import TfRecordSpecParser

"""Holds additional information about/from Dataset parsing.

Attributes:
    filenames_placeholder: A placeholder for Dataset file inputs.
    num_features: Number of features available in the Dataset.
"""
DatasetContext = namedtuple("DatasetContext", ["filenames",
                                               "feature_names",
                                               "multispec_feature_groups"])


class Datasets(object):
    @classmethod
    def get_featran_example_dataset(cls,
                                    dir_path,
                                    feature_mapping_fn=None,
                                    num_threads=mp.cpu_count(),
                                    num_threads_per_file=mp.cpu_count(),
                                    block_length=32,
                                    tf_record_spec_path=None):
        """Get `Dataset` of parsed `Example` protos.

        Args:
            dir_path: Directory path containing features.
            feature_mapping_fn: A function which maps `FeatureInfo` to `FixedLenFeature` or
                `VarLenFeature` values. Default maps all features to
                tf.FixedLenFeature((), feature_info.type, default_value=0).
            num_threads: A `tf.int32` scalar or `tf.Tensor`, represents number of files to process
                concurrently.
            num_threads_per_file: A `tf.int32` scalar or `tf.Tensor`, represents number of threads
                used concurrently per file.
            block_length: A `tf.int32` scalar or `tf.Tensor`, represents buffer size for results
                from any of the parsing threads.
            tf_record_spec_path: Filepath to feature description file. Default is
                `_tf_record_spec.json` inside `dir_path`.

        Returns:
            A Tuple of two elements: (dataset, dataset_context). First element is a `Dataset`, which
            holds results of the parsing of `Example` protos. Second element holds a
            `DatasetContext` (see doc of `DatasetContext`).
        """

        filenames = cls.__get_tfrecord_filenames(dir_path)
        dataset = tf.data.Dataset.from_tensor_slices(filenames)

        feature_info, compression, feature_groups = TfRecordSpecParser.parse_tf_record_spec(
            tf_record_spec_path, dir_path)

        feature_mapping_fn = feature_mapping_fn or cls.__get_default_feature_mapping_fn
        features = feature_mapping_fn(feature_info)

        def _parse_function(example_proto):
            return tf.parse_single_example(example_proto, features)

        dataset = cls.__try_enable_sharding(dataset)

        # TODO(rav): does `map` need to be inside `interleave`, what are the performance diff?
        dataset = dataset.interleave(lambda f: tf.data.TFRecordDataset(f, compression),
                                     cycle_length=num_threads,
                                     block_length=block_length)
        dataset = dataset.map(_parse_function, num_parallel_calls=num_threads_per_file)
        context = DatasetContext(filenames, features, feature_groups)
        return dataset, context

    @staticmethod
    def __get_tfrecord_filenames(dir_path):
        assert isinstance(dir_path, str), "dir_path is not a String: %r" % dir_path
        assert file_io.file_exists(dir_path), "directory `%s` does not exist" % dir_path
        assert file_io.is_directory(dir_path), "`%s` is not a directory" % dir_path
        from os.path import join as pjoin
        flist = file_io.list_directory(dir_path)
        input_files = [pjoin(dir_path, x) for x in filter(lambda f: not f.startswith("_"), flist)]
        filenames = tf.placeholder_with_default(input_files, shape=[None])
        return filenames

    @staticmethod
    def __get_default_feature_mapping_fn(feature_info):
        fm = [(f.name, tf.FixedLenFeature((), f.type, default_value=0)) for f in feature_info]
        return dict(fm)

    @staticmethod
    def __try_enable_sharding(dataset):
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

    @staticmethod
    def mk_dataset_training(training_data_dir, feature_mapping_fn):
        """Make a training `Dataset`.

        Args:
            training_data_dir: a directory contains training data.
            feature_mapping_fn: A function which maps `FeatureInfo` to `FixedLenFeature` or
                `VarLenFeature` values. Default maps all features to
                tf.FixedLenFeature((), feature_info.type, default_value=0).

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
            feature_mapping_fn: A function which maps `FeatureInfo` to `FixedLenFeature` or
                `VarLenFeature` values. Default maps all features to
                tf.FixedLenFeature((), feature_info.type, default_value=0).

        Returns:
            A `Dataset` that should be used for evaluation purposes.
        """
        with tf.name_scope("evaluation-input"):
            eval_dataset, _ = Datasets.get_featran_example_dataset(
                eval_data_dir,
                feature_mapping_fn=feature_mapping_fn)
            return eval_dataset
