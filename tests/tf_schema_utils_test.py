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

import os
from tempfile import NamedTemporaryFile

import tensorflow as tf
from tensorflow.python.platform import test
from spotify_tensorflow.tf_schema_utils import feature_spec_to_schema, \
    schema_to_feature_spec, schema_txt_file_to_feature_spec  # noqa: E501


class TfSchemaUtilsTest(test.TestCase):
    def test_round_trip(self):
        feature_spec = {
            "scalar_feature_1": tf.FixedLenFeature(shape=[], dtype=tf.int64),
            "scalar_feature_2": tf.FixedLenFeature(shape=[], dtype=tf.int64),
            "scalar_feature_3": tf.FixedLenFeature(shape=[], dtype=tf.float32),
            "varlen_feature_1": tf.VarLenFeature(dtype=tf.float32),
            "varlen_feature_2": tf.VarLenFeature(dtype=tf.string),
            "1d_vector_feature": tf.FixedLenFeature(shape=[1], dtype=tf.string),
            "2d_vector_feature": tf.FixedLenFeature(shape=[2, 2], dtype=tf.float32),
            "sparse_feature": tf.SparseFeature("idx", "value", tf.float32, 10),
        }
        inferred_schema = feature_spec_to_schema(feature_spec)
        inferred_feature_spec = schema_to_feature_spec(inferred_schema)
        self.assertEqual(inferred_feature_spec, feature_spec)

    def test_schema_txt_to_feature_spec(self):
        schema_txt = """
            feature {
                name: "test_feature"
                value_count {
                    min: 1
                    max: 1
                }
                type: FLOAT
                presence {
                    min_count: 1
                }
            }
        """

        with NamedTemporaryFile() as f:
            f.write(schema_txt)
            f.flush()
            os.fsync(f)
            feature_spec = schema_txt_file_to_feature_spec(f.name)
            self.assertEqual(feature_spec, {"test_feature": tf.VarLenFeature(dtype=tf.float32)})
