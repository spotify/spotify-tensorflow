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
from typing import Tuple, Union, Dict, Iterator  # noqa: F401

import six
import numpy as np  # noqa: F401
import pandas as pd
import tensorflow as tf
from tensorflow_metadata.proto.v0.schema_pb2 import Schema  # noqa: F401


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Datasets(object):

    @staticmethod
    def _assert_eager(endpoint):
        assert tf.executing_eagerly(), "Eager execution is required for a %s endpoint! " \
                                       "Add this add the begining of your main:\n\nimport " \
                                       "tensorflow as tf\ntf.enable_eager_execution()\n\n" % \
                                       endpoint

    @classmethod
    def parse_schema(cls, schema_path):
        # type: (str) -> Tuple[Dict[str, Union[tf.FixedLenFeature, tf.VarLenFeature, tf.SparseFeature]], Schema]  # noqa: E501
        """
        Returns TensorFlow Feature Spec and parsed tf.metadata Schema for given tf.metadata Schema.

        :param schema_path: tf.metadata Schema path
        """
        from .tf_schema_utils import SchemaToFeatureSpec
        schema = SchemaToFeatureSpec.parse_schema_file(schema_path)
        return SchemaToFeatureSpec.apply(schema), schema

    @classmethod
    def examples_via_schema(cls,
                            file_pattern,  # type: str
                            schema_path,  # type: str
                            compression_type=None,  # type: str
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
        # type: (...) -> tf.data.Dataset
        """Get `Dataset` of parsed `Example` protos.

        :param file_pattern: List of files or patterns of file paths containing
                             `Example` records. See `tf.gfile.Glob` for pattern rules
        :param schema_path: tf.metadata Schema path
        :param compression_type: TFRecord compression type, see `tf.data.TFRecordDataset` doc
        :param batch_size: see `tensorflow.contrib.data.make_batched_features_dataset` doc
        :param shuffle: see `tensorflow.contrib.data.make_batched_features_dataset` doc
        :param num_epochs: see `tensorflow.contrib.data.make_batched_features_dataset` doc
        :param shuffle_buffer_size: see `tensorflow.contrib.data.make_batched_features_dataset` doc
        :param shuffle_seed: see `tensorflow.contrib.data.make_batched_features_dataset` doc
        :param prefetch_buffer_size: see `tensorflow.contrib.data.make_batched_features_dataset` doc
        :param reader_num_threads: see `tensorflow.contrib.data.make_batched_features_dataset` doc
        :param parser_num_threads: see `tensorflow.contrib.data.make_batched_features_dataset` doc
        :param sloppy_ordering: see `tensorflow.contrib.data.make_batched_features_dataset` doc
        :param drop_final_batch: see `tensorflow.contrib.data.make_batched_features_dataset` doc

        :return `Dataset`, which holds results of the parsing of `Example` protos
        """
        return cls._examples(file_pattern,
                             schema_path=schema_path,
                             compression_type=compression_type,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_epochs=num_epochs,
                             shuffle_buffer_size=shuffle_buffer_size,
                             shuffle_seed=shuffle_seed,
                             prefetch_buffer_size=prefetch_buffer_size,
                             reader_num_threads=reader_num_threads,
                             parser_num_threads=parser_num_threads,
                             sloppy_ordering=sloppy_ordering,
                             drop_final_batch=drop_final_batch)

    @classmethod
    def examples_via_feature_spec(cls,
                                  file_pattern,  # type: str
                                  feature_spec,  # type: Dict[str, Union[tf.FixedLenFeature, tf.VarLenFeature, tf.SparseFeature]]  # noqa: E501
                                  compression_type=None,  # type: str
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
        # type: (...) -> tf.data.Dataset
        """Get `Dataset` of parsed `Example` protos.

        :param file_pattern: List of files or patterns of file paths containing
                             `Example` records. See `tf.gfile.Glob` for pattern rules
        :param feature_spec: TensorFlow feature spec
        :param compression_type: TFRecord compression type, see `tf.data.TFRecordDataset` doc
        :param batch_size: see `tensorflow.contrib.data.make_batched_features_dataset` doc
        :param shuffle: see `tensorflow.contrib.data.make_batched_features_dataset` doc
        :param num_epochs: see `tensorflow.contrib.data.make_batched_features_dataset` doc
        :param shuffle_buffer_size: see `tensorflow.contrib.data.make_batched_features_dataset` doc
        :param shuffle_seed: see `tensorflow.contrib.data.make_batched_features_dataset` doc
        :param prefetch_buffer_size: see `tensorflow.contrib.data.make_batched_features_dataset` doc
        :param reader_num_threads: see `tensorflow.contrib.data.make_batched_features_dataset` doc
        :param parser_num_threads: see `tensorflow.contrib.data.make_batched_features_dataset` doc
        :param sloppy_ordering: see `tensorflow.contrib.data.make_batched_features_dataset` doc
        :param drop_final_batch: see `tensorflow.contrib.data.make_batched_features_dataset` doc

        :return `Dataset`, which holds results of the parsing of `Example` protos
        """
        return cls._examples(file_pattern,
                             feature_spec=feature_spec,
                             compression_type=compression_type,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_epochs=num_epochs,
                             shuffle_buffer_size=shuffle_buffer_size,
                             shuffle_seed=shuffle_seed,
                             prefetch_buffer_size=prefetch_buffer_size,
                             reader_num_threads=reader_num_threads,
                             parser_num_threads=parser_num_threads,
                             sloppy_ordering=sloppy_ordering,
                             drop_final_batch=drop_final_batch)

    @classmethod
    def _examples(cls,
                  file_pattern,  # type: str
                  schema_path=None,  # type: str
                  feature_spec=None,  # type: Dict[str, Union[tf.FixedLenFeature, tf.VarLenFeature, tf.SparseFeature]]  # noqa: E501
                  compression_type=None,  # type: str
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
        # type: (...) -> tf.data.Dataset

        if schema_path:
            feature_spec, _ = cls.parse_schema(schema_path)

        logger.debug("Will parse features from: `%s`, using features spec: %s",
                     file_pattern,
                     str(feature_spec))

        from tensorflow.contrib.data import make_batched_features_dataset
        reader_args = [compression_type] if compression_type else None
        dataset = make_batched_features_dataset(file_pattern,
                                                batch_size=batch_size,
                                                features=feature_spec,
                                                reader_args=reader_args,
                                                num_epochs=num_epochs,
                                                shuffle=shuffle,
                                                shuffle_buffer_size=shuffle_buffer_size,
                                                shuffle_seed=shuffle_seed,
                                                prefetch_buffer_size=prefetch_buffer_size,
                                                reader_num_threads=reader_num_threads,
                                                parser_num_threads=parser_num_threads,
                                                sloppy_ordering=sloppy_ordering,
                                                drop_final_batch=drop_final_batch)
        return dataset

    class __DictionaryEndpoint(object):

        @classmethod
        def examples_via_schema(cls,
                                file_pattern,  # type: str
                                schema_path,  # type: str
                                default_value=0,  # type: float
                                batch_size=128,  # type: int
                                compression_type=None,  # type: str
                                shuffle=True,  # type: bool
                                num_epochs=1,  # type: int
                                shuffle_buffer_size=10000,  # type: int
                                shuffle_seed=42,  # type: int
                                prefetch_buffer_size=1,  # type: int
                                reader_num_threads=1,  # type: int
                                parser_num_threads=2,  # type: int
                                sloppy_ordering=False,  # type: bool
                                drop_final_batch=False  # type: bool
                                ):
            # type: (...) -> Iterator[Dict[str, np.ndarray]]
            """
            Read a TF dataset and load it into a dictionary of NumPy Arrays.

            :param file_pattern: List of files or patterns of file paths containing
                                 `Example` records. See `tf.gfile.Glob` for pattern rules
            :param default_value: Value used if a sparse feature is missing.
            :param schema_path: tf.metadata Schema path
            :param compression_type: TFRecord compression type, see `tf.data.TFRecordDataset` doc
            :param batch_size: batch size, set to the size of the dataset to read all data at once
            :param shuffle: see `tensorflow.contrib.data.make_batched_features_dataset` doc
            :param num_epochs: see `tensorflow.contrib.data.make_batched_features_dataset` doc
            :param shuffle_buffer_size: see `tensorflow.contrib.data.make_batched_features_dataset`
                                        doc
            :param shuffle_seed: see `tensorflow.contrib.data.make_batched_features_dataset` doc
            :param prefetch_buffer_size: see `tensorflow.contrib.data.make_batched_features_dataset`
                                         doc
            :param reader_num_threads: see `tensorflow.contrib.data.make_batched_features_dataset`
                                       doc
            :param parser_num_threads: see `tensorflow.contrib.data.make_batched_features_dataset`
                                       doc
            :param sloppy_ordering: see `tensorflow.contrib.data.make_batched_features_dataset` doc
            :param drop_final_batch: see `tensorflow.contrib.data.make_batched_features_dataset` doc

            :return Dictionary of NumPy arrays
            """
            return cls._examples(file_pattern=file_pattern,
                                 schema_path=schema_path,
                                 default_value=default_value,
                                 batch_size=batch_size,
                                 compression_type=compression_type,
                                 shuffle=shuffle,
                                 num_epochs=num_epochs,
                                 shuffle_buffer_size=shuffle_buffer_size,
                                 shuffle_seed=shuffle_seed,
                                 prefetch_buffer_size=prefetch_buffer_size,
                                 reader_num_threads=reader_num_threads,
                                 parser_num_threads=parser_num_threads,
                                 sloppy_ordering=sloppy_ordering,
                                 drop_final_batch=drop_final_batch)

        @classmethod
        def examples_via_feature_spec(cls,
                                      file_pattern,  # type: str
                                      feature_spec,  # type: Dict[str, Union[tf.FixedLenFeature, tf.VarLenFeature, tf.SparseFeature]]  # noqa: E501
                                      default_value=0,  # type: float
                                      batch_size=128,  # type: int
                                      compression_type=None,  # type: str
                                      shuffle=True,  # type: bool
                                      num_epochs=1,  # type: int
                                      shuffle_buffer_size=10000,  # type: int
                                      shuffle_seed=42,  # type: int
                                      prefetch_buffer_size=1,  # type: int
                                      reader_num_threads=1,  # type: int
                                      parser_num_threads=2,  # type: int
                                      sloppy_ordering=False,  # type: bool
                                      drop_final_batch=False  # type: bool
                                      ):
            # type: (...) -> Iterator[Dict[str, np.ndarray]]
            """
            Read a TF dataset and load it into a dictionary of NumPy Arrays.

            :param file_pattern: List of files or patterns of file paths containing
                                 `Example` records. See `tf.gfile.Glob` for pattern rules
            :param feature_spec: TensorFlow feature spec
            :param default_value: Value used if a sparse feature is missing.
            :param compression_type: TFRecord compression type, see `tf.data.TFRecordDataset` doc
            :param batch_size: batch size, set to the size of the dataset to read all data at once
            :param shuffle: see `tensorflow.contrib.data.make_batched_features_dataset` doc
            :param num_epochs: see `tensorflow.contrib.data.make_batched_features_dataset` doc
            :param shuffle_buffer_size: see `tensorflow.contrib.data.make_batched_features_dataset`
                                        doc
            :param shuffle_seed: see `tensorflow.contrib.data.make_batched_features_dataset` doc
            :param prefetch_buffer_size: see `tensorflow.contrib.data.make_batched_features_dataset`
                                         doc
            :param reader_num_threads: see `tensorflow.contrib.data.make_batched_features_dataset`
                                       doc
            :param parser_num_threads: see `tensorflow.contrib.data.make_batched_features_dataset`
                                       doc
            :param sloppy_ordering: see `tensorflow.contrib.data.make_batched_features_dataset` doc
            :param drop_final_batch: see `tensorflow.contrib.data.make_batched_features_dataset` doc

            :return Dictionary of NumPy arrays
            """
            return cls._examples(file_pattern=file_pattern,
                                 feature_spec=feature_spec,
                                 default_value=default_value,
                                 batch_size=batch_size,
                                 compression_type=compression_type,
                                 shuffle=shuffle,
                                 num_epochs=num_epochs,
                                 shuffle_buffer_size=shuffle_buffer_size,
                                 shuffle_seed=shuffle_seed,
                                 prefetch_buffer_size=prefetch_buffer_size,
                                 reader_num_threads=reader_num_threads,
                                 parser_num_threads=parser_num_threads,
                                 sloppy_ordering=sloppy_ordering,
                                 drop_final_batch=drop_final_batch)

        @classmethod
        def _examples(cls,
                      file_pattern,  # type: str
                      schema_path=None,  # type: str
                      feature_spec=None,  # type: Dict[str, Union[tf.FixedLenFeature, tf.VarLenFeature, tf.SparseFeature]]  # noqa: E501
                      default_value=0,  # type: float
                      compression_type=None,  # type: str
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
            # type: (...) -> Iterator[Dict[str, np.ndarray]]
            Datasets._assert_eager("Dictionary")

            def get_numpy(tensor):
                if isinstance(tensor, tf.Tensor):
                    return tensor.numpy()
                elif isinstance(tensor, tf.SparseTensor):
                    # If it's a SparseTensor, which is the representation of VarLenFeature and
                    # SparseFeature, we convert it to dense representation, and further is it's
                    # a scalar, we reshape to to a vector

                    shape = tensor.dense_shape.numpy()
                    # first element is batch size
                    if shape[1] == 0:
                        # this feature is not defined for any of the examples in the batch
                        return np.repeat(default_value, shape[0])

                    numpy_dense = tf.sparse_tensor_to_dense(tensor,
                                                            default_value=default_value).numpy()
                    if shape[1] == 1:
                        # this is scalar feature, reshape to a vector
                        return numpy_dense.reshape(shape[0])
                    else:
                        return numpy_dense
                else:
                    raise ValueError("This type %s is not supported!", type(tensor).__name__)

            dataset = Datasets._examples(file_pattern=file_pattern,
                                         schema_path=schema_path,
                                         feature_spec=feature_spec,
                                         compression_type=compression_type,
                                         batch_size=batch_size,
                                         shuffle=shuffle,
                                         num_epochs=num_epochs,
                                         shuffle_buffer_size=shuffle_buffer_size,
                                         shuffle_seed=shuffle_seed,
                                         prefetch_buffer_size=prefetch_buffer_size,
                                         reader_num_threads=reader_num_threads,
                                         parser_num_threads=parser_num_threads,
                                         sloppy_ordering=sloppy_ordering,
                                         drop_final_batch=drop_final_batch)
            for batch in dataset:
                yield {name: get_numpy(eager_tensor) for name, eager_tensor in six.iteritems(batch)}

    dict = __DictionaryEndpoint()

    class __DataFrameEndpoint(object):

        @classmethod
        def examples_via_schema(cls,
                                file_pattern,  # type: str
                                schema_path,  # type: str
                                default_value=0,  # type: float
                                batch_size=128,  # type: int
                                compression_type=None,  # type: str
                                shuffle=True,  # type: bool
                                num_epochs=1,  # type: int
                                shuffle_buffer_size=10000,  # type: int
                                shuffle_seed=42,  # type: int
                                prefetch_buffer_size=1,  # type: int
                                reader_num_threads=1,  # type: int
                                parser_num_threads=2,  # type: int
                                sloppy_ordering=False,  # type: bool
                                drop_final_batch=False  # type: bool
                                ):
            # type: (...) -> Iterator[pd.DataFrame]
            """
            Read a TF dataset in batches, each one yields a Pandas DataFrame.

            :param file_pattern: List of files or patterns of file paths containing
                                 `Example` records. See `tf.gfile.Glob` for pattern rules
            :param schema_path: tf.metadata Schema path
            :param default_value: Value used if a sparse feature is missing.
            :param batch_size: batch size, set to the size of the dataset to read all data at once
            :param compression_type: TFRecord compression type, see `tf.data.TFRecordDataset` doc
            :param shuffle: see `tensorflow.contrib.data.make_batched_features_dataset` doc
            :param num_epochs: see `tensorflow.contrib.data.make_batched_features_dataset` doc
            :param shuffle_buffer_size: see `tensorflow.contrib.data.make_batched_features_dataset`
                                        doc
            :param shuffle_seed: see `tensorflow.contrib.data.make_batched_features_dataset` doc
            :param prefetch_buffer_size: see `tensorflow.contrib.data.make_batched_features_dataset`
                                         doc
            :param reader_num_threads: see `tensorflow.contrib.data.make_batched_features_dataset`
                                       doc
            :param parser_num_threads: see `tensorflow.contrib.data.make_batched_features_dataset`
                                       doc
            :param sloppy_ordering: see `tensorflow.contrib.data.make_batched_features_dataset` doc
            :param drop_final_batch: see `tensorflow.contrib.data.make_batched_features_dataset` doc

            :return A Python Generator, yielding batches of data in a Pandas DataFrame
            """
            return cls._examples(file_pattern=file_pattern,
                                 schema_path=schema_path,
                                 default_value=default_value,
                                 batch_size=batch_size,
                                 compression_type=compression_type,
                                 shuffle=shuffle,
                                 num_epochs=num_epochs,
                                 shuffle_buffer_size=shuffle_buffer_size,
                                 shuffle_seed=shuffle_seed,
                                 prefetch_buffer_size=prefetch_buffer_size,
                                 reader_num_threads=reader_num_threads,
                                 parser_num_threads=parser_num_threads,
                                 sloppy_ordering=sloppy_ordering,
                                 drop_final_batch=drop_final_batch)

        @classmethod
        def examples_via_feature_spec(cls,
                                      file_pattern,  # type: str
                                      feature_spec,  # type: Dict[str, Union[tf.FixedLenFeature, tf.VarLenFeature, tf.SparseFeature]]  # noqa: E501
                                      default_value=0,  # type: float
                                      batch_size=128,  # type: int
                                      compression_type=None,  # type: str
                                      shuffle=True,  # type: bool
                                      num_epochs=1,  # type: int
                                      shuffle_buffer_size=10000,  # type: int
                                      shuffle_seed=42,  # type: int
                                      prefetch_buffer_size=1,  # type: int
                                      reader_num_threads=1,  # type: int
                                      parser_num_threads=2,  # type: int
                                      sloppy_ordering=False,  # type: bool
                                      drop_final_batch=False  # type: bool
                                      ):
            # type: (...) -> Iterator[pd.DataFrame]
            """
            Read a TF dataset in batches, each one yields a Pandas DataFrame.

            :param file_pattern: List of files or patterns of file paths containing
                                 `Example` records. See `tf.gfile.Glob` for pattern rules
            :param feature_spec: TensorFlow feature spec
            :param default_value: Value used if a sparse feature is missing.
            :param batch_size: batch size, set to the size of the dataset to read all data at once
            :param compression_type: TFRecord compression type, see `tf.data.TFRecordDataset` doc
            :param shuffle: see `tensorflow.contrib.data.make_batched_features_dataset` doc
            :param num_epochs: see `tensorflow.contrib.data.make_batched_features_dataset` doc
            :param shuffle_buffer_size: see `tensorflow.contrib.data.make_batched_features_dataset`
                                        doc
            :param shuffle_seed: see `tensorflow.contrib.data.make_batched_features_dataset` doc
            :param prefetch_buffer_size: see `tensorflow.contrib.data.make_batched_features_dataset`
                                         doc
            :param reader_num_threads: see `tensorflow.contrib.data.make_batched_features_dataset`
                                       doc
            :param parser_num_threads: see `tensorflow.contrib.data.make_batched_features_dataset`
                                       doc
            :param sloppy_ordering: see `tensorflow.contrib.data.make_batched_features_dataset` doc
            :param drop_final_batch: see `tensorflow.contrib.data.make_batched_features_dataset` doc

            :return A Python Generator, yielding batches of data in a Pandas DataFrame
            """
            return cls._examples(file_pattern=file_pattern,
                                 feature_spec=feature_spec,
                                 default_value=default_value,
                                 batch_size=batch_size,
                                 compression_type=compression_type,
                                 shuffle=shuffle,
                                 num_epochs=num_epochs,
                                 shuffle_buffer_size=shuffle_buffer_size,
                                 shuffle_seed=shuffle_seed,
                                 prefetch_buffer_size=prefetch_buffer_size,
                                 reader_num_threads=reader_num_threads,
                                 parser_num_threads=parser_num_threads,
                                 sloppy_ordering=sloppy_ordering,
                                 drop_final_batch=drop_final_batch)

        @classmethod
        def _examples(cls,
                      file_pattern,  # type: str
                      schema_path=None,  # type: str
                      feature_spec=None,  # type: Dict[str, Union[tf.FixedLenFeature, tf.VarLenFeature, tf.SparseFeature]]  # noqa: E501
                      default_value=0,  # type: float
                      compression_type=None,  # type: str
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
            # type: (...) -> Iterator[pd.DataFrame]
            Datasets._assert_eager("DataFrame")
            dataset = Datasets.dict._examples(file_pattern=file_pattern,
                                              schema_path=schema_path,
                                              default_value=default_value,
                                              feature_spec=feature_spec,
                                              compression_type=compression_type,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_epochs=num_epochs,
                                              shuffle_buffer_size=shuffle_buffer_size,
                                              shuffle_seed=shuffle_seed,
                                              prefetch_buffer_size=prefetch_buffer_size,
                                              reader_num_threads=reader_num_threads,
                                              parser_num_threads=parser_num_threads,
                                              sloppy_ordering=sloppy_ordering,
                                              drop_final_batch=drop_final_batch)
            for d in dataset:
                yield pd.DataFrame(data=d)

    dataframe = __DataFrameEndpoint()
