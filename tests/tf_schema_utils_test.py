#!/usr/bin/env python
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

import hashlib
import urllib2

import tensorflow as tf
from spotify_tensorflow.tf_schema_utils import SchemaToFeatureSpec, FeatureSpecToSchema


class TFSchemaUtilsTest(tf.test.TestCase):

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
        inferred_schema = FeatureSpecToSchema.apply(feature_spec)
        inferred_feature_spec = SchemaToFeatureSpec.apply(inferred_schema)
        self.assertEqual(inferred_feature_spec, feature_spec)

    def test_metadata_schema_sha(self):
        """
        Test the upstream tf_metadata schema didn't change since compiled.

        if changed, you can update:
        ```
        curl -O https://raw.githubusercontent.com/tensorflow/metadata/master/tensorflow_metadata/proto/v0/schema.proto
        protoc --python_out=spotify_tensorflow/tensorflow_metadata/proto/v0 schema.proto
        ```
        """  # noqa: E501
        schema_url = "https://raw.githubusercontent.com/tensorflow/metadata/master/tensorflow_metadata/proto/v0/schema.proto"  # noqa: E501
        response = urllib2.urlopen(schema_url)
        schema_txt = response.read()
        actual_sha = hashlib.sha256(schema_txt).hexdigest()
        expected_sha = "fac5de9b802bd3e481606e2bb63496d169efda1cb90c7f848d95bc9d4823172b"
        self.assertEqual(expected_sha, actual_sha)
