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

import argparse
import errno
import os
import sys

import tensorflow as tf
from tensorflow.python.lib.io import file_io
from spotify_tensorflow.example_decoders import ExampleWithFeatureSpecDecoder, ExampleDecoder
from spotify_tensorflow.tf_schema_utils import SchemaToFeatureSpec


def get_args():
    parser = argparse.ArgumentParser(description="Output TFRecords as JSON")
    parser.add_argument("-s", "--schema", help="Path to Schema protobuf file. Uses Example if not "
                                               "supplied.")
    parser.add_argument("tf_records_paths",
                        metavar="TF_RECORDS_PATH",
                        nargs="+",
                        help="TFRecords file (or directory containing .tfrecords files)")
    return parser.parse_args()


def resolve_dir_schemas(dirs):
    dir_schemas = []
    for d in dirs:
        s = os.path.join(d, "_schema.pb")
        s_inferred = os.path.join(d, "_inferred_schema.pb")
        if file_io.file_exists(s):
            dir_schemas.append((d, s))
        elif file_io.file_exists(s_inferred):
            dir_schemas.append((d, s_inferred))
        else:
            dir_schemas.append((d, None))
    return dict(dir_schemas)


def list_tf_records(paths, default_schema):
    for p in paths:
        if file_io.file_exists(p):
            files = [p]
        else:
            # dir or glob
            files = [f for f in file_io.get_matching_files(p) if f.endswith(".tfrecords")]

        dirs = set(os.path.dirname(f) for f in files)
        if default_schema:
            dir_schemas = dict((d, default_schema) for d in dirs)
        else:
            dir_schemas = resolve_dir_schemas(dirs)

        for f in files:
            yield f, dir_schemas[os.path.dirname(f)]


def main():
    cmdline_args = get_args()

    default_schema = None
    if cmdline_args.schema:
        default_schema = cmdline_args.schema
        assert file_io.file_exists(default_schema), "File not found: {}".format(default_schema)

    last_schema = None
    example_decoder = ExampleDecoder()

    for tf_record_file, schema in list_tf_records(cmdline_args.tf_records_paths,
                                                  default_schema):
        assert file_io.file_exists(tf_record_file), "File not found: {}".format(tf_record_file)

        # Load the right example decoder
        if schema != last_schema:
            last_schema = schema
            if schema:
                schema_object = SchemaToFeatureSpec.parse_schema_file(schema)
                feature_spec = SchemaToFeatureSpec.apply(schema_object)
                example_decoder = ExampleWithFeatureSpecDecoder(feature_spec)
            else:
                example_decoder = ExampleDecoder()

        # decode the examples
        for record in tf.python_io.tf_record_iterator(tf_record_file):
            try:
                print(example_decoder.to_json(record))
            except IOError as e:
                if e.errno == errno.EPIPE:
                    sys.exit(0)
                raise e


if __name__ == "__main__":
    main()
