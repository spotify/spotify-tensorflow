# -*- coding: utf-8 -*-
#
#  Copyright 2018 Spotify AB.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.python.lib.io import file_io
from spotify_tensorflow.tensorflow_metadata.proto.v0.schema_pb2 import INT, FLOAT, BYTES, Schema


class TFTypeMapper(object):
    """
    Helps to map between TensorFlow DTypes and tf.metadata Schema types (and back).
    """

    # FIXME: TF DTypes are only partially supported:
    # https://www.tensorflow.org/api_docs/python/tf/DType
    _PB_TF_TYPES = {
        INT: [tf.int32, tf.int64],
        FLOAT: [tf.float32, tf.float64],
        BYTES: [tf.string, tf.bool]
    }

    def __init__(self):
        tf_to_pb_array = []
        for p, ts in self._PB_TF_TYPES.items():
            for t in ts:
                tf_to_pb_array.append((t, p))
        self._tf_to_pb = dict(tf_to_pb_array)
        self._int_domain_to_dtype = dict([((t.min, t.max), t) for t in self._PB_TF_TYPES[INT]])
        self._float_domain_to_dtype = dict([((t.min, t.max), t) for t in self._PB_TF_TYPES[FLOAT]])

    def proto_to_tf_type(self, feature, is_sparse=False):
        """
        Go from tf.metadata Schema type to TensorFlow DTypes.
        """
        proto_type = feature.type
        if proto_type == BYTES:
            if feature.HasField("bool_domain"):
                return tf.bool
            return tf.string
        if proto_type == INT:
            k = self._extract_domain_min_max(feature, is_sparse)
            if k in self._int_domain_to_dtype:
                return self._int_domain_to_dtype[k]
            return tf.int64  # default
        if proto_type == FLOAT:
            k = self._extract_domain_min_max(feature, is_sparse)
            if k in self._float_domain_to_dtype:
                return self._float_domain_to_dtype[k]
            return tf.float32  # default
        raise Exception("Proto type not supported " + proto_type)

    def tf_to_proto_type(self, tf_type):
        """
        Go from TensorFlow DTypes to tf.metadata Schema type.
        """
        return self._tf_to_pb[tf_type]

    @staticmethod
    def _extract_domain_min_max(feature, is_sparse=False):
        proto_type = feature.type
        if is_sparse:
            if proto_type == INT:
                d = feature.domain.ints
            elif proto_type == FLOAT:
                d = feature.domain.floats
            else:
                raise Exception("Proto type not supported " + proto_type)
        else:
            if proto_type == INT:
                d = feature.int_domain
            elif proto_type == FLOAT:
                d = feature.float_domain
            else:
                raise Exception("Proto type not supported " + proto_type)
        return d.min, d.max


class SchemaToFeatureSpec:
    """
    Convert from a tf.metadata Schema to a TensorFlow feature_spec object.
    """
    _tf_type_mapper = TFTypeMapper()

    @classmethod
    def apply(cls, schema):
        """
        Main entry point.
        """
        decoded_feature_spec = dict(map(cls._parse_dense_feature, schema.feature))
        decoded_feature_spec.update(dict(map(cls._parse_sparse_feature, schema.sparse_feature)))
        return decoded_feature_spec.copy()

    @classmethod
    def _parse_dense_feature(cls, feature):
        dtype = cls._tf_type_mapper.proto_to_tf_type(feature)
        if feature.HasField("shape"):
            shape = [d.size for d in feature.shape.dim if d.HasField("size")]
            return feature.name, tf.FixedLenFeature(shape=shape,
                                                    dtype=dtype)
        else:
            return feature.name, tf.VarLenFeature(dtype=dtype)

    @classmethod
    def _parse_sparse_feature(cls, feature):
        if len(feature.index_feature) == 1:
            index_key = feature.index_feature[0].name
        else:
            index_key = [idf.name for idf in feature.index_feature]
        dtype = cls._tf_type_mapper.proto_to_tf_type(feature, is_sparse=True)
        if len(feature.dense_shape.dim) == 1:
            size = feature.dense_shape.dim[0].size
        else:
            size = [d.size for d in feature.dense_shape.dim]
        return feature.name, tf.SparseFeature(index_key=index_key,
                                              value_key=feature.value_feature.name,
                                              dtype=dtype,
                                              size=size)

    @staticmethod
    def parse_schema_file(schema_path):
        """
        Read a schema file and return the proto object.
        """
        assert file_io.file_exists(schema_path), "File not found: {}".format(schema_path)
        schema = Schema()
        with file_io.FileIO(schema_path, "r") as f:
            schema.ParseFromString(f.read())
        return schema


class FeatureSpecToSchema(object):
    """
    Convert from a TensorFlow feature_spec object to a tf.metadata Schema.
    """
    _tf_type_mapper = TFTypeMapper()

    @classmethod
    def apply(cls, feature_spec):
        """
        Main entry point.
        """
        schema_proto = Schema()
        for k, v in feature_spec.items():
            if isinstance(v, tf.SparseFeature):
                cls._add_sparse_feature_to_proto(schema_proto, k, v)
            else:
                cls._add_feature_to_proto(schema_proto, k, v)

        return schema_proto

    @classmethod
    def _add_feature_to_proto(cls, schema_proto, feature_name, feature_val):
        f = schema_proto.feature.add()
        f.name = feature_name

        if hasattr(feature_val, "shape"):
            # fixlen features
            fixed_shape = f.shape

            if len(feature_val.shape) == 0:
                fixed_shape.dim.add()
            else:
                for s in feature_val.shape:
                    dim = fixed_shape.dim.add()
                    dim.size = s

        f.type = cls._tf_type_mapper.tf_to_proto_type(feature_val.dtype)

        if f.type == INT:
            f.int_domain.min = feature_val.dtype.min
            f.int_domain.max = feature_val.dtype.max
        if f.type == FLOAT:
            f.float_domain.min = feature_val.dtype.min
            f.float_domain.max = feature_val.dtype.max

    @classmethod
    def _add_sparse_feature_to_proto(cls, schema_proto, feature_name, feature_val):
        f = schema_proto.sparse_feature.add()
        f.name = feature_name

        fv = f.value_feature
        fv.name = feature_val.value_key

        f.type = cls._tf_type_mapper.tf_to_proto_type(feature_val.dtype)

        fixed_shape = f.dense_shape
        if isinstance(feature_val.index_key, list):
            for index_name, f in zip(feature_val.index_key, feature_val.size):
                idf = f.index_feature.add()
                idf.name = index_name

                dim = fixed_shape.dim.add()
                dim.size = f
        else:
            idf = f.index_feature.add()
            idf.name = feature_val.index_key

            dim = fixed_shape.dim.add()
            dim.size = feature_val.size

        if f.type == INT:
            f.domain.ints.min = feature_val.dtype.min
            f.domain.ints.max = feature_val.dtype.max
        if f.type == FLOAT:
            f.domain.floats.min = feature_val.dtype.min
            f.domain.floats.max = feature_val.dtype.max
