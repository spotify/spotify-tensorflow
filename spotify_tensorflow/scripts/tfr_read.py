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

import argparse
import errno
import os
import sys

import tensorflow as tf
from tensorflow.python.lib.io import file_io
from spotify_tensorflow.example_decoders import ExampleWithFeatureSpecDecoder, ExampleDecoder
from spotify_tensorflow.tf_schema_utils import schema_file_to_feature_spec


def resolve_schema(dir, default_schema=None):
    if default_schema is not None:
        return default_schema

    for schema_file_name in ["_schema.pb", "_inferred_schema.pb"]:
        s = os.path.join(dir, schema_file_name)
        if file_io.file_exists(s):
            return s


def list_tf_records(paths, default_schema):
    for p in paths:
        files = [f for f in file_io.get_matching_files(p) if f.endswith(".tfrecords")]
        if len(files) == 0:
            raise Exception("Couldn't find any .tfrecords file in path or glob [{}]".format(p))
        for f in files:
            yield f, resolve_schema(os.path.dirname(f), default_schema)


def get_decoder_from_schema(schema):
    if schema is None:
        return ExampleDecoder()
    else:
        feature_spec = schema_file_to_feature_spec(schema)
        return ExampleWithFeatureSpecDecoder(feature_spec)


def tfr_read_to_json(tf_records_paths, schema_path=None):
    if schema_path is not None:
        assert file_io.file_exists(schema_path), "File not found: {}".format(schema_path)

    for tf_record_file, schema in list_tf_records(tf_records_paths, schema_path):
        assert file_io.file_exists(tf_record_file), "File not found: {}".format(tf_record_file)

        decoder = get_decoder_from_schema(schema)
        for record in tf.python_io.tf_record_iterator(tf_record_file):
            yield decoder.to_json(record)


def main():
    parser = argparse.ArgumentParser(description="Output TFRecords as JSON")
    parser.add_argument("-s", "--schema", help="Path to Schema protobuf file. Uses Example if not "
                                               "supplied.")
    parser.add_argument("tf_records_paths",
                        metavar="TF_RECORDS_PATH",
                        nargs="+",
                        help="TFRecords file (or directory containing .tfrecords files)")

    args = parser.parse_args()

    try:
        for json_str in tfr_read_to_json(args.tf_records_paths, args.schema):
            print(json_str)
    except IOError as e:
        if e.errno == errno.EPIPE:
            sys.exit(0)
        raise e


if __name__ == "__main__":
    main()
