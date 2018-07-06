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

import tensorflow as tf
from tensorflow.core.example import example_pb2
from tensorflow.python.lib.io import file_io
from tensorflow_metadata.proto.v0.schema_pb2 import Schema
from tensorflow_metadata.proto.v0.schema_pb2 import INT, FLOAT, BYTES
from tensorflow_transform.coders import example_proto_coder
from tensorflow_transform.tf_metadata import dataset_schema
import json
import numpy as np
from google.protobuf.json_format import MessageToJson

to_tf_type = {INT: tf.int32,
              FLOAT: tf.float32,
              BYTES: tf.string}


class ExampleDecoder:
    def to_json(self, example_str):
        ex = example_pb2.Example()
        ex.ParseFromString(example_str)
        return MessageToJson(ex)


class ExampleWithSchemaDecoder:

    def __init__(self, schema_path):
        assert file_io.file_exists(schema_path), "File not found: {}".format(schema_path)
        schema = dataset_schema.from_feature_spec(
            self._schema_proto_to_feature_spec(schema_path)
        )
        self._coder = example_proto_coder.ExampleProtoCoder(schema)

    def to_json(self, example_str):
        decoded = self._coder.decode(example_str)
        decoded_json = json.dumps(decoded, cls=self._NumpyEncoder)
        return decoded_json

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    @staticmethod
    def _feature_to_feature_spec(feature):
        if len(feature.shape.dim) != 0:
            return feature.name, tf.FixedLenFeature(shape=tuple(d.size for d in feature.shape.dim),
                                                    dtype=to_tf_type[feature.type])
        else:
            return feature.name, tf.VarLenFeature(dtype=to_tf_type[feature.type])

    @staticmethod
    def _feature_to_sparse_feature_spec(sf):
        if len(sf.index_feature) == 1:
            index_key = sf.index_feature[0].name
        else:
            index_key = [idf.name for idf in sf.index_feature]
        return sf.name, tf.SparseFeature(index_key=index_key,
                                         value_key=sf.value_feature.name,
                                         dtype=to_tf_type[sf.type],
                                         size=[d.size for d in sf.dense_shape.dim])

    @classmethod
    def _schema_proto_to_feature_spec(cls, schema_file):
        schema = Schema()
        with file_io.FileIO(schema_file, "r") as f:
            schema.ParseFromString(f.read())

        name_to_feature = dict(map(cls._feature_to_feature_spec, schema.feature))
        name_to_feature.update(dict(map(cls._feature_to_sparse_feature_spec, schema.sparse_feature)))
        return name_to_feature.copy()