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

from __future__ import unicode_literals

import json

import tensorflow as tf
from google.protobuf import text_format  # type: ignore
from spotify_tensorflow.example_decoders import ExampleDecoder, ExampleWithFeatureSpecDecoder


class ExampleDecodersTest(tf.test.TestCase):

    @property
    def example_str(self):
        example = tf.train.Example()
        text_format.Merge("""
        features {
          feature { key: "scalar_feature_1" value { int64_list { value: [ 12 ] } } }
          feature { key: "varlen_feature_1"
                    value { float_list { value: [ 89.0 ] } } }
          feature { key: "scalar_feature_2" value { int64_list { value: [ 12 ] } } }
          feature { key: "scalar_feature_3"
                    value { float_list { value: [ 1.0 ] } } }
          feature { key: "1d_vector_feature"
                    value { bytes_list { value: [ 'this is a ,text' ] } } }
          feature { key: "2d_vector_feature"
                    value { float_list { value: [ 1.0, 2.0, 3.0, 4.0 ] } } }
          feature { key: "varlen_feature_2"
                    value { bytes_list { value: [ 'female' ] } } }
          feature { key: "sparse_feature_value" value { float_list { value: [ 12.0, 20.0 ] } } }
          feature { key: "sparse_feature_idx" value { int64_list { value: [ 1, 4 ] } } }
        }
        """, example)
        return example.SerializeToString()

    def test_example_with_feature_spec_decoder(self):
        feature_spec = {
            "scalar_feature_1": tf.FixedLenFeature(shape=[], dtype=tf.int64),
            "scalar_feature_2": tf.FixedLenFeature(shape=[], dtype=tf.int64),
            "scalar_feature_3": tf.FixedLenFeature(shape=[], dtype=tf.float32),
            "varlen_feature_1": tf.VarLenFeature(dtype=tf.float32),
            "varlen_feature_2": tf.VarLenFeature(dtype=tf.string),
            "1d_vector_feature": tf.FixedLenFeature(shape=[1], dtype=tf.string),
            "2d_vector_feature": tf.FixedLenFeature(shape=[2, 2], dtype=tf.float32),
            "sparse_feature": tf.SparseFeature("sparse_feature_idx", "sparse_feature_value",
                                               tf.float32, 10),
        }

        dec = ExampleWithFeatureSpecDecoder(feature_spec)
        actual_json = json.loads(dec.to_json(self.example_str))
        expected_decoded = {
            "scalar_feature_1": 12,
            "scalar_feature_2": 12,
            "scalar_feature_3": 1.0,
            "varlen_feature_1": [89.0],
            "1d_vector_feature": ["this is a ,text"],
            "2d_vector_feature": [[1.0, 2.0], [3.0, 4.0]],
            "varlen_feature_2": ["female"],
            "sparse_feature_idx": [1, 4],
            "sparse_feature_value": [12.0, 20.0],
        }
        self.assertEqual(actual_json, expected_decoded)

    def test_example_decoder(self):
        dec = ExampleDecoder()
        actual_json = json.loads(dec.to_json(self.example_str))
        expected_json = json.loads("""
        {
          "features": {
            "feature": {
              "1d_vector_feature": {
                "bytesList": {
                  "value": [
                    "dGhpcyBpcyBhICx0ZXh0"
                  ]
                }
              },
              "varlen_feature_2": {
                "bytesList": {
                  "value": [
                    "ZmVtYWxl"
                  ]
                }
              },
              "sparse_feature_idx": {
                "int64List": {
                  "value": [
                    "1",
                    "4"
                  ]
                }
              },
              "varlen_feature_1": {
                "floatList": {
                  "value": [
                    89.0
                  ]
                }
              },
              "scalar_feature_2": {
                "int64List": {
                  "value": [
                    "12"
                  ]
                }
              },
              "scalar_feature_3": {
                "floatList": {
                  "value": [
                    1.0
                  ]
                }
              },
              "scalar_feature_1": {
                "int64List": {
                  "value": [
                    "12"
                  ]
                }
              },
              "sparse_feature_value": {
                "floatList": {
                  "value": [
                    12.0,
                    20.0
                  ]
                }
              },
              "2d_vector_feature": {
                "floatList": {
                  "value": [
                    1.0,
                    2.0,
                    3.0,
                    4.0
                  ]
                }
              }
            }
          }
        }
        """)
        self.assertEqual(expected_json, actual_json)
