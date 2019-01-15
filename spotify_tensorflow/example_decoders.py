# -*- coding: utf-8 -*-
#
# Copyright 2017-2019 Spotify AB.
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

from __future__ import absolute_import, division, print_function

import json

import numpy as np
from google.protobuf.json_format import MessageToJson
from tensorflow.core.example import example_pb2
from tensorflow_transform.coders import example_proto_coder
from tensorflow_transform.tf_metadata import dataset_schema


class ExampleDecoder(object):
    """
    Decode a tf.Example payload using the example.proto schema
    """
    def to_json(self, example_str):  # type: (str) -> str
        """
        Converts a single tf.Example to JSon a string
        :param example_str: tf.Example payload
        """
        ex = example_pb2.Example()
        ex.ParseFromString(example_str)
        return MessageToJson(ex)


class ExampleWithFeatureSpecDecoder(ExampleDecoder):
    """
    Decode a tf.Example payload using a TensorFlow feature_spec
    """
    def __init__(self, feature_spec):
        super(ExampleWithFeatureSpecDecoder, self).__init__()
        schema = dataset_schema.from_feature_spec(feature_spec)
        self._coder = example_proto_coder.ExampleProtoCoder(schema)

    class _NumpyArrayEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, bytes):
                return obj.decode()
            return json.JSONEncoder.default(self, obj)

    def to_json(self, example_str):  # type: (str) -> str
        """
        Converts a single tf.Example to Json a string
        :param example_str: tf.Example payload
        """
        decoded = self._coder.decode(example_str)
        decoded_json = json.dumps(decoded, cls=self._NumpyArrayEncoder)
        return decoded_json
