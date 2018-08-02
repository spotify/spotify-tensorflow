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
from collections import namedtuple, OrderedDict
from os.path import join as pjoin
from typing import Callable, Tuple, Union, Dict, List, Iterator, Optional  # noqa: F401
import timeit

import six
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.lib.io import file_io

from .tf_record_spec_parser import TfRecordSpecParser, FeatureInfo  # noqa: F401

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DatasetContext(namedtuple("DatasetContext", ["filenames",
                                                   "features",
                                                   "multispec_feature_groups"])):
    """Holds additional information about/from Dataset parsing.

    Attributes:
        filenames: A placeholder for Dataset file inputs.
        features: Map features and their type available in the Dataset.
        multispec_feature_groups: Feature names, grouped as they appear in Featran MultiFeatureSpec.
    """
    # TODO: change to something more type friendly than namedtuple


class Datasets(object):

    @classmethod
    def get_featran_example_dataset(cls,
                                    dir_path,  # type: str
                                    feature_mapping_fn=None,  # type: Callable[[List[FeatureInfo]], OrderedDict[str, Union[tf.FixedLenFeature, tf.VarLenFeature]]]  # noqa: E501
                                    tf_record_spec_path=None,  # type: str
                                    batch_size=128,  # type: int
                                    shuffle=True,  # type: bool
                                    num_epochs=1,  # type: int
                                    shuffle_buffer_size=10000,  # type: int
                                    shuffle_seed=None,  # type: int
                                    prefetch_buffer_size=1,  # type: int
                                    reader_num_threads=1,  # type: int
                                    parser_num_threads=2,  # type: int
                                    sloppy_ordering=False,  # type: bool
                                    drop_final_batch=False  # type: bool
                                    ):
        # type: (...) -> Tuple[tf.data.Dataset, DatasetContext]
        """Get `Dataset` of parsed `Example` protos.

        Args:
            dir_path: Directory path containing features.
            feature_mapping_fn: Feature mapping function, default maps all features to
                                tf.FixedLenFeature((), tf.int64, default_value=0).
            tf_record_spec_path: Filepath to feature description file. Default is
                                 `_tf_record_spec.json` inside `dir_path`.
            batch_size: see tensorflow.contrib.data.make_batched_features_dataset doc
            shuffle: see tensorflow.contrib.data.make_batched_features_dataset doc
            num_epochs: see tensorflow.contrib.data.make_batched_features_dataset doc
            shuffle_buffer_size: see tensorflow.contrib.data.make_batched_features_dataset doc
            shuffle_seed: see tensorflow.contrib.data.make_batched_features_dataset doc
            prefetch_buffer_size: see tensorflow.contrib.data.make_batched_features_dataset doc
            reader_num_threads: see tensorflow.contrib.data.make_batched_features_dataset doc
            parser_num_threads: see tensorflow.contrib.data.make_batched_features_dataset doc
            sloppy_ordering: see tensorflow.contrib.data.make_batched_features_dataset doc
            drop_final_batch: see tensorflow.contrib.data.make_batched_features_dataset doc

        Returns:
            A Tuple of two elements: (dataset, dataset_context). First element is a `Dataset`, which
            holds results of the parsing of `Example` protos. Second element holds a
            `DatasetContext` (see doc of `DatasetContext`).
        """
        _, filenames = cls._get_tfrecord_filenames(dir_path)

        feature_info, compression, feature_groups = TfRecordSpecParser.parse_tf_record_spec(
            tf_record_spec_path, dir_path)

        feature_mapping_fn = feature_mapping_fn or cls._get_default_feature_mapping_fn
        features = feature_mapping_fn(feature_info)

        from tensorflow.contrib.data import make_batched_features_dataset
        dataset = make_batched_features_dataset(filenames,
                                                batch_size=batch_size,
                                                features=features,
                                                reader_args=[compression],
                                                num_epochs=num_epochs,
                                                shuffle=shuffle,
                                                shuffle_buffer_size=shuffle_buffer_size,
                                                shuffle_seed=shuffle_seed,
                                                prefetch_buffer_size=prefetch_buffer_size,
                                                reader_num_threads=reader_num_threads,
                                                parser_num_threads=parser_num_threads,
                                                sloppy_ordering=sloppy_ordering,
                                                drop_final_batch=drop_final_batch)
        return dataset, DatasetContext(filenames, features, feature_groups)

    @staticmethod
    def _get_tfrecord_filenames(dir_path):
        # type: (str) -> Tuple[tf.Tensor, List[str]]
        assert isinstance(dir_path, str), "dir_path is not a String: %r" % dir_path
        assert file_io.file_exists(dir_path), "directory `%s` does not exist" % dir_path
        assert file_io.is_directory(dir_path), "`%s` is not a directory" % dir_path
        flist = file_io.list_directory(dir_path)
        input_files = [pjoin(dir_path, x) for x in filter(lambda f: not f.startswith("_"), flist)]
        filenames = tf.placeholder_with_default(input_files, shape=[None])
        return filenames, input_files

    @staticmethod
    def _get_default_feature_mapping_fn(feature_info):
        # type: (List[FeatureInfo]) -> OrderedDict[str, Union[tf.FixedLenFeature, tf.VarLenFeature]]
        fm = [(f.name, tf.FixedLenFeature([], f.kind)) for f in feature_info]
        return OrderedDict(fm)

    @classmethod
    def get_context(cls,
                    dir_path,  # type: str
                    feature_mapping_fn=None,  # type: Callable[[List[FeatureInfo]], OrderedDict[str, Union[tf.FixedLenFeature, tf.VarLenFeature]]]  # noqa: E501
                    tf_record_spec_path=None  # type: str
                    ):
        # type: (...) -> DatasetContext
        """Return Featran's record spec context.

        Args:
            dir_path: Directory path containing features.
            feature_mapping_fn: Feature mapping function, default maps all features to
                                tf.FixedLenFeature((), tf.int64, default_value=0).
            tf_record_spec_path: Filepath to feature description file. Default is
                                 `_tf_record_spec.json` inside `dir_path`.

        Returns:
            Returns `DatasetContext` (see doc of `DatasetContext`) for given dataset.
        """
        filenames, _ = cls._get_tfrecord_filenames(dir_path)
        feature_info, _, feature_groups = TfRecordSpecParser.parse_tf_record_spec(
                tf_record_spec_path, dir_path)
        feature_mapping_fn = feature_mapping_fn or cls._get_default_feature_mapping_fn
        features = feature_mapping_fn(feature_info)
        return DatasetContext(filenames, features, feature_groups)

    @classmethod
    def mk_iter(cls,
                data_dir,  # type: str
                scope="tfrecords_iter",  # type: str
                feature_mapping_fn=None,  # type: Callable[[List[FeatureInfo]], OrderedDict[str, Union[tf.FixedLenFeature, tf.VarLenFeature]]]  # noqa: E501
                mk_iterator_fn=None,  # type: Callable[[tf.data.Dataset], tf.data.Iterator]
                batch_size=128,  # type: int
                shuffle=True,  # type: bool
                shuffle_buffer_size=10000,  # type: int
                shuffle_seed=None,  # type: int
                prefetch_buffer_size=1,  # type: int
                take_count=-1  # type: int
                ):
        # type: (...) ->  Tuple[tf.data.Iterator, DatasetContext]
        """Make a training `Dataset` iterator.

        Args:
            data_dir: a directory contains training data.
            scope: TF scope for this op (e.g 'training-input').
            feature_mapping_fn: Feature mapping function, default maps all features to
                                tf.FixedLenFeature((), tf.int64, default_value=0).
            mk_iterator_fn: `Dataset` iterator to use. By default `make_one_shot_iterator()` is
                            used.

        Returns:
            A `Dataset` iterator that should be used for training purposes and a `DatasetContext`
            object.
        """
        with tf.name_scope(scope):
            dataset, context = cls.get_featran_example_dataset(data_dir,
                                                               feature_mapping_fn,
                                                               batch_size=batch_size,
                                                               prefetch_buffer_size=prefetch_buffer_size,  # noqa: E501
                                                               shuffle=shuffle,
                                                               shuffle_buffer_size=shuffle_buffer_size,  # noqa: E501
                                                               shuffle_seed=shuffle_seed)
            dataset = dataset.take(take_count)

            mk_iterator_fn = mk_iterator_fn or cls._mk_one_shot_iterator
            return mk_iterator_fn(dataset), context

    @staticmethod
    def _mk_one_shot_iterator(dataset):  # type: (tf.data.Dataset) -> tf.data.Iterator
        return dataset.make_one_shot_iterator()

    class __DictionaryEndpoint(object):
        @classmethod
        def read_dataset(cls,
                         dataset_path,  # type: str
                         take=sys.maxsize,  # type: int
                         feature_mapping_fn=None  # type: Callable[[List[FeatureInfo]], OrderedDict[str, Union[tf.FixedLenFeature, tf.VarLenFeature]]]  # noqa: E501
                         ):
            # type: (...) -> Dict[str, np.ndarray]
            """
            Read a TF dataset and load it into a Dictionary of Numpy Arrays.

            :param dataset_path: Path to the TF Records Dataset
            :param take: Number of records to read when building the MultiFeatureSpec in Featran
            :param feature_mapping_fn: Override the TF record reading function
            :return: A Dictionary containing the dataset
            """
            return six.next(cls.batch_iterator(dataset_path, take, feature_mapping_fn))

        @classmethod
        def batch_iterator(cls,
                           dataset_path,  # type: str
                           batch_size=10000,  # type: int
                           feature_mapping_fn=None  # type: Callable[[List[FeatureInfo]], OrderedDict[str, Union[tf.FixedLenFeature, tf.VarLenFeature]]]  # noqa: E501
                           ):
            # type: (...) -> Iterator[Dict[str, np.ndarray]]
            """
            Read a TF dataset in batches, each one yielded as a Dictonary.

            :param dataset_path: Path to the TF Records Dataset
            :param batch_size: Size of each batches when building the MultiFeatureSpec in Featran
            :param feature_mapping_fn: Override the TF record reading function
            :return: A Python Generator, yielding batches of data in a Dictionary
            """
            training_it, context = Datasets.mk_iter(
                dataset_path,
                feature_mapping_fn=feature_mapping_fn)

            for batch in cls.__FeatureGenerator(training_it, batch_size, context):
                yield batch

        class __FeatureGenerator(object):
            def __init__(self,
                         training_it,  # type: tf.data.Iterator
                         batch_size,  # type: int
                         context  # type: DatasetContext
                         ):
                # type: (...) -> None
                self.batch_size = batch_size
                self.batch_iter = training_it.get_next()
                self.context = context
                self.buff = OrderedDict()  # type: OrderedDict[str, np.ndarray]

            def __iter__(self):
                # type: () -> Iterator[OrderedDict[str, np.ndarray]]
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
                # type: () -> int
                if len(self.buff.keys()) == 0:
                    return 0
                else:
                    return len(self.buff[list(self.buff.keys())[0]])

            def _append(self, v1, v2):
                # type: (np.ndarray, np.ndarray) -> np.ndarray
                if type(v1) is np.ndarray:
                    if v1.ndim == 1:
                        return np.append(v1, v2)
                    elif v1.ndim == 2:
                        return np.vstack([v1, v2])
                    else:
                        raise ValueError("Only 1 or 2 dimensional features are supported")
                else:
                    # TODO: what case does this code path support?
                    return v1.append(v2)

            def _get_batch(self, sess):
                # type: (tf.Session) -> OrderedDict[str, np.ndarray]
                while self._get_buff_size() < self.batch_size:
                    t = timeit.default_timer()
                    current_batch = sess.run(self.batch_iter)
                    for k in self.context.features.keys():
                        if k in self.buff:
                            self.buff[k] = self._append(self.buff[k], current_batch[k])
                        else:
                            self.buff[k] = current_batch[k]
                    logger.info("Fetched %d / %s records (%4d TFExamples/s)" % (
                        self._get_buff_size(),
                        str(self.batch_size) if self.batch_size < sys.maxsize else "?",
                        self.batch_size / (timeit.default_timer() - t)))
                ret = OrderedDict()  # type: OrderedDict[str, np.ndarray]
                for k in list(self.context.features.keys()):
                    ret[k] = self.buff[k][:self.batch_size]
                    self.buff[k] = self.buff[k][self.batch_size:]

                return ret

    dict = __DictionaryEndpoint()

    class __DataFrameEndpoint(object):
        @classmethod
        def read_dataset(cls,
                         dataset_path,  # type: str
                         take=sys.maxsize,  # type: int
                         unpack_multispec=False,  # type: bool
                         feature_mapping_fn=None  # type: Callable[[List[FeatureInfo]], OrderedDict[str, Union[tf.FixedLenFeature, tf.VarLenFeature]]]  # noqa: E501
                         ):
            # type: (...) -> pd.DataFrame
            """
            Read a TF dataset and load it into a Pandas DataFrame.

            :param dataset_path: Path to the TF Records Dataset
            :param take: Number of records to read
            :param unpack_multispec: Returns an array of DataFrames, order is the same when
                                     building the MultiFeatureSpec in Featran
            :param feature_mapping_fn: Override the TF record reading function
            :return: A Pandas DataFrame containing the dataset
            """
            return six.next(cls.batch_iterator(dataset_path, take, unpack_multispec,
                                               feature_mapping_fn))

        @classmethod
        def batch_iterator(cls,
                           dataset_path,  # type: str
                           batch_size=10000,  # type: int
                           unpack_multispec=False,  # type: bool
                           feature_mapping_fn=None  # type: Callable[[List[FeatureInfo]], OrderedDict[str, Union[tf.FixedLenFeature, tf.VarLenFeature]]]  # noqa: E501
                           ):
            # type: (...) -> Union[Iterator[pd.DataFrame], Iterator[List[pd.DataFrame]]]
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
            # type: (Dict[str, np.ndarray], Optional[List[List[str]]]) -> Union[pd.DataFrame, List[pd.DataFrame]]  # noqa: E501
            df = pd.DataFrame(batch)
            if not multispec_feature_groups:
                return df[list(batch.keys())]
            return [df[f] for f in multispec_feature_groups]

    dataframe = __DataFrameEndpoint()
