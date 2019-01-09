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

from typing import Union, Dict  # noqa: F401

import google.protobuf.text_format
import tensorflow as tf  # noqa: F401
from tensorflow.python.lib.io import file_io
from tensorflow_metadata.proto.v0.schema_pb2 import Schema
from tensorflow_transform.tf_metadata import schema_utils


def feature_spec_to_schema(feature_spec):
    # type: (Dict[str, Union[tf.FixedLenFeature, tf.VarLenFeature, tf.SparseFeature]]) -> Schema
    """
    Convert a Tensorflow feature_spec object to a tf.metadata Schema.
    """
    return schema_utils.schema_from_feature_spec(feature_spec)


def parse_schema_file(schema_path):  # type: (str) -> Schema
    """
    Read a schema file and return the proto object.
    """
    assert file_io.file_exists(schema_path), "File not found: {}".format(schema_path)
    schema = Schema()
    with file_io.FileIO(schema_path, "rb") as f:
        schema.ParseFromString(f.read())
    return schema


def parse_schema_txt_file(schema_path):  # type: (str) -> Schema
    """
    Parse a tf.metadata Schema txt file into its in-memory representation.
    """
    assert file_io.file_exists(schema_path), "File not found: {}".format(schema_path)
    schema = Schema()
    schema_text = file_io.read_file_to_string(schema_path)
    google.protobuf.text_format.Parse(schema_text, schema)
    return schema


def schema_to_feature_spec(schema):
    # type: (Schema) -> Dict[str, Union[tf.FixedLenFeature, tf.VarLenFeature, tf.SparseFeature]]
    """
    Convert a tf.metadata Schema to a Tensorflow feature_spec object.
    """
    return schema_utils.schema_as_feature_spec(schema).feature_spec


def schema_file_to_feature_spec(schema_path):
    # type: (str) -> Dict[str, Union[tf.FixedLenFeature, tf.VarLenFeature, tf.SparseFeature]]
    """
    Convert a serialized tf.metadata Schema file to a Tensorflow feature_spec object
    """
    schema = parse_schema_file(schema_path)
    return schema_to_feature_spec(schema)


def schema_txt_file_to_feature_spec(schema_path):
    # type: (str) -> Dict[str, Union[tf.FixedLenFeature, tf.VarLenFeature, tf.SparseFeature]]
    """
    Convert a tf.metadata Schema text file to a TensorFlow feature_spec object.
    """
    schema = parse_schema_txt_file(schema_path)
    return schema_to_feature_spec(schema)
