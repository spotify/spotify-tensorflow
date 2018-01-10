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

import json
from collections import defaultdict, namedtuple

import tensorflow as tf
from tensorflow.python.lib.io import file_io

FeatureInfo = namedtuple("FeatureInfo", ["name", "kind", "tags"])


class TfRecordSpecParser(object):
    @classmethod
    def parse_tf_record_spec(cls, tf_record_desc_path, dir_path):
        tf_record_spec_path = cls.__get_tf_record_spec_path(tf_record_desc_path, dir_path)
        with file_io.FileIO(tf_record_spec_path, "r") as f:
            spec = json.load(f)
        assert spec["version"] == 1

        # features types
        type_map = {
            "FloatList": tf.float32,
            "Int64List": tf.int64,
            "BytesList": tf.int8
        }
        feature_info = [FeatureInfo(fi["name"], type_map[fi["kind"]], fi["tags"])
                        for fi in spec["features"]]
        assert len(feature_info) > 0

        # groups by multispec
        multispec_feature_groups = []
        if "multispec-id" in feature_info[0][2]:
            d = defaultdict(set)
            for name, _, tags in feature_info:
                d[int(tags["multispec-id"])].add(name)
            multispec_feature_groups = [None] * len(d)
            for i, f in d.iteritems():
                multispec_feature_groups[i] = list(f)

        # parse compression
        compression_map = {
            "UNCOMPRESSED": "",
            "DEFLATE": "ZLIB",
            "GZIP": "GZIP"
        }
        assert spec["compression"] in compression_map, \
            "Compression %s not supported by TF." % spec["compression"]
        return feature_info, compression_map[spec["compression"]], multispec_feature_groups

    @staticmethod
    def __get_tf_record_spec_path(tf_record_desc_path, dir_path):
        if tf_record_desc_path is not None:
            assert isinstance(tf_record_desc_path, str), \
                "tf_record_desc_path is not a String: %r" % tf_record_desc_path
            assert file_io.file_exists(tf_record_desc_path), \
                "feature desc `%s` does not exist" % tf_record_desc_path
            return tf_record_desc_path
        assert isinstance(dir_path, str), "dir_path is not a String: %r" % dir_path
        assert file_io.file_exists(dir_path), "directory `%s` does not exist" % dir_path
        assert file_io.is_directory(dir_path), "`%s` is not a directory" % dir_path
        from os.path import join as pjoin
        default_tf_record_spec_filename = "_tf_record_spec.json"
        return pjoin(dir_path, default_tf_record_spec_filename)
