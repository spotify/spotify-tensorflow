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

import logging
import sys
import json
import os
from collections import namedtuple, OrderedDict
from os.path import join as pjoin
import timeit

import six
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.lib.io import file_io

from .tf_record_spec_parser import TfRecordSpecParser

__all__ = ["Datasets"]

FLAGS = tf.flags.FLAGS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DatasetContext(namedtuple("DatasetContext", ["filenames",
                                                   "features",
                                                   "multispec_feature_groups"])):
    """Holds additional information about/from Dataset parsing.

    Attributes:
        filenames: A placeholder for Dataset file inputs.
        features: Map features and their type available in the Dataset.
        multispec_feature_groups: Feature names, grouped as they appear in Featran MultiFeatureSpec
    """


class Datasets(object):
    @classmethod
    def _get_featran_example_dataset(cls,
                                     dir_path,
                                     feature_mapping_fn=None,
                                     tf_record_spec_path=None):
        """Get `Dataset` of parsed `Example` protos.

        Args:
            dir_path: Directory path containing features.
            feature_mapping_fn: A function which maps feature names to `FixedLenFeature` or
                `VarLenFeature` values. Default maps all features to
                tf.FixedLenFeature((), tf.int64, default_value=0).
            tf_record_spec_path: Filepath to feature description file. Default is
                `_tf_record_spec.json` inside `dir_path`.

        Returns:
            A Tuple of two elements: (dataset, dataset_context). First element is a `Dataset`, which
            holds results of the parsing of `Example` protos. Second element holds a
            `DatasetContext` (see doc of `DatasetContext`).
        """
        filenames = cls._get_tfrecord_filenames(dir_path)

        feature_info, compression, feature_groups = TfRecordSpecParser.parse_tf_record_spec(
            tf_record_spec_path, dir_path)

        if FLAGS.interleaving_threads > 0:
            dataset = tf.data.Dataset.from_tensor_slices(filenames)
            dataset = dataset.interleave(lambda f: tf.data.TFRecordDataset(f, compression),
                                         cycle_length=FLAGS.interleaving_threads,
                                         block_length=FLAGS.interleaving_block_length)
        else:
            dataset = tf.data.TFRecordDataset(filenames, compression)

        dataset = cls._try_enable_sharding(dataset)

        feature_mapping_fn = feature_mapping_fn or cls._get_default_feature_mapping_fn
        features = feature_mapping_fn(feature_info)

        def _parse_function(example_proto):
            return tf.parse_single_example(example_proto, features)

        dataset = dataset.map(_parse_function, num_parallel_calls=FLAGS.parsing_threads)
        return dataset, DatasetContext(filenames, features, feature_groups)

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
    def _get_default_feature_mapping_fn(feature_info):
        fm = [(f.name, tf.FixedLenFeature((), f.kind, default_value=0)) for f in feature_info]
        return OrderedDict(fm)

    @staticmethod
    def _try_enable_sharding(dataset):
        tf_config = os.environ.get("TF_CONFIG")

        if tf_config is not None:
            tf_config_json = json.loads(tf_config)
            tf.logging.info("Found TF_CONFIG: %s" % tf_config)
            num_workers = len(tf_config_json.get("cluster", {}).get("worker", []))
            worker_index = tf_config_json.get("task", {}).get("index", None)
            if num_workers > 0:
                tf.logging.info("Sharding dataset on worker_index=%s out of %s workers"
                                % (worker_index, num_workers))
                return dataset.shard(num_workers, worker_index)
        return dataset

    @classmethod
    def mk_iter(cls, data_dir,
                scope="tfrecords_iter",
                feature_mapping_fn=None,
                mk_iterator_fn=None):
        """Make a training `Dataset` iterator.

        Args:
            data_dir: a directory contains training data.
            scope: TF scope for this op (e.g 'training-input').
            feature_mapping_fn: A function which maps `FeatureInfo` to `FixedLenFeature` or
                `VarLenFeature` values. Default maps all features to
                tf.FixedLenFeature((), feature_info.type, default_value=0).
            mk_iterator_fn: `Dataset` iterator to use. By default `make_one_shot_iterator()` is
                used.

        Returns:
            A `Dataset` iterator that should be used for training purposes and a `DatasetContext`
            object.
        """
        with tf.name_scope(scope):
            dataset, context = cls._get_featran_example_dataset(data_dir,
                                                                feature_mapping_fn)
            if FLAGS.shuffle_buffer_size > 0:
                dataset = dataset.shuffle(FLAGS.shuffle_buffer_size)

            if FLAGS.batch_size > 0:
                dataset = dataset.batch(FLAGS.batch_size)

            dataset = dataset.take(FLAGS.take_count)

            if FLAGS.prefetch_buffer_size > 0:
                dataset = dataset.prefetch(FLAGS.prefetch_buffer_size)

            mk_iterator_fn = mk_iterator_fn or cls._mk_one_shot_iterator
            return mk_iterator_fn(dataset), context

    @staticmethod
    def _mk_one_shot_iterator(dataset):
        return dataset.make_one_shot_iterator()

    class __DictionaryEndpoint(object):
        @classmethod
        def read_dataset(cls, dataset_path, take=sys.maxsize, feature_mapping_fn=None):
            """
            Read a TF dataset and load it into a Dictionary of Numpy Arrays.

            :param dataset_path: Path to the TF Records Dataset
            :param take: Number of records to read
            when building the MultiFeatureSpec in Featran
            :param feature_mapping_fn: Override the TF record reading function
            :return: A Dictionary containing the dataset
            """
            return six.next(cls.batch_iterator(dataset_path, take, feature_mapping_fn))

        @classmethod
        def batch_iterator(cls, dataset_path, batch_size=10000, feature_mapping_fn=None):
            """
            Read a TF dataset in batches, each one yielded as a Dictonary.

            :param dataset_path: Path to the TF Records Dataset
            :param batch_size: Size of each batches
            when building the MultiFeatureSpec in Featran
            :param feature_mapping_fn: Override the TF record reading function
            :return: A Python Generator, yielding batches of data in a Dictionary
            """
            training_it, context = Datasets.mk_iter(
                dataset_path,
                feature_mapping_fn=feature_mapping_fn)

            for batch in cls.__FeatureGenerator(training_it, batch_size, context):
                yield batch

        class __FeatureGenerator(object):
            def __init__(self, training_it, batch_size, context):
                self.batch_size = batch_size
                self.batch_iter = training_it.get_next()
                self.context = context
                self.buff = None

            def __iter__(self):
                logger.info("Starting TF Session...")
                with tf.Session() as sess:
                    logger.info("Reading TFRecords...")
                    while True:
                        try:
                            yield self._get_batch(sess)
                        except tf.errors.OutOfRangeError:
                            logger.info("End of dataset.")
                            break
                yield self.buff

            def _get_buff_size(self):
                if self.buff is None:
                    return 0
                else:
                    return len(self.buff[list(self.buff.keys())[0]])

            def _append(self, v1, v2):
                if type(v1) is np.ndarray:
                    if(v1.ndim == 1):
                        return np.append(v1, v2)
                    elif(v1.ndim == 2):
                        return np.vstack([v1, v2])
                    else:
                        raise ValueError("Only 1 or 2 dimensional features are supported")
                else:
                    return v1.append(v2)

            def _get_batch(self, sess):
                if self.buff is None:
                    self.buff = OrderedDict()
                    first_result = sess.run(self.batch_iter)
                    for k in self.context.features.keys():
                        self.buff[k] = first_result[k]
                while self._get_buff_size() < self.batch_size:
                    t = timeit.default_timer()
                    current_batch = sess.run(self.batch_iter)
                    for k in self.context.features.keys():
                        self.buff[k] = self._append(self.buff[k], current_batch[k])
                    logger.info("Fetched %d / %s records (%4d TFExamples/s)" % (
                        self._get_buff_size(),
                        str(self.batch_size) if self.batch_size < sys.maxsize else "?",
                        FLAGS.batch_size / (timeit.default_timer() - t)))
                ret = OrderedDict()
                for k in list(self.context.features.keys()):
                    ret[k] = self.buff[k][:self.batch_size]
                    self.buff[k] = self.buff[k][self.batch_size:]

                return ret

    dict = __DictionaryEndpoint()

    class __DataFrameEndpoint(object):
        @classmethod
        def read_dataset(cls, dataset_path,
                         take=sys.maxsize,
                         unpack_multispec=False,
                         feature_mapping_fn=None):
            """
            Read a TF dataset and load it into a Pandas DataFrame.

            :param dataset_path: Path to the TF Records Dataset
            :param take: Number of records to read
            :param unpack_multispec: Returns an array of DataFrames, order is the same
            when building the MultiFeatureSpec in Featran
            :param feature_mapping_fn: Override the TF record reading function
            :return: A Pandas DataFrame containing the dataset
            """
            return six.next(cls.batch_iterator(dataset_path, take, unpack_multispec,
                                               feature_mapping_fn))

        @classmethod
        def batch_iterator(cls, dataset_path,
                           batch_size=10000,
                           unpack_multispec=False,
                           feature_mapping_fn=None):
            """
            Read a TF dataset in batches, each one yielded as a Pandas DataFrame.

            :param dataset_path: Path to the TF Records Dataset
            :param batch_size: Size of each batches
            :param unpack_multispec: Returns an array of DataFrames, order is the same
            when building the MultiFeatureSpec in Featran
            :param feature_mapping_fn: Override the TF record reading function
            :return: A Python Generator, yielding batches of data in a Pandas DataFrame
            """
            training_it, context = Datasets.mk_iter(
                dataset_path,
                feature_mapping_fn=feature_mapping_fn)

            groups = context.multispec_feature_groups if unpack_multispec else None
            for batch in Datasets.dict.batch_iterator(dataset_path, batch_size, feature_mapping_fn):
                yield cls.__format_df(batch, groups)

        @staticmethod
        def __format_df(batch, multispec_feature_groups):
            df = pd.DataFrame(batch)
            if not multispec_feature_groups:
                print("TYPE", type(batch))
                return df[list(batch.keys())]
            return [df[f] for f in multispec_feature_groups]

        @staticmethod
        def __to_df(feature_dict):
            ret = OrderedDict()
            for k in feature_dict:
                ret[k] = feature_dict[k].tolist()
            print(ret)
            return ret

    dataframe = __DataFrameEndpoint()
