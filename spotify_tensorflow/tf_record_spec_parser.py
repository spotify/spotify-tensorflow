# -*- coding: utf-8 -*-
#
# Copyright 2018 Spotify AB.
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
from collections import OrderedDict, namedtuple
from typing import Tuple, Text, List  # noqa: F401

import six
import tensorflow as tf
from tensorflow.python.lib.io import file_io

FeatureInfo = namedtuple("FeatureInfo", ["name", "kind", "tags"])


class TfRecordSpecParser(object):
    # TODO: make this private, or handle arguments better, having both of the required doesn't make sense # noqa: E501
    @classmethod
    def parse_tf_record_spec(cls,
                             tf_record_desc_path,  # type: str
                             dir_path  # type: str
                             ):
        # type: (...) -> Tuple[List[FeatureInfo], str, List[List[str]]]
        """
        Parses TF record spec saved by Scio. tf_record_desc_path takes precedence over dir_path.

        :param tf_record_desc_path: fully qualified path to TF record spec - this must be a single
                                    file.
        :param dir_path: path of the data directory, will use default spec file name
        :return:
        """
        tf_record_spec_path = cls.__get_tf_record_spec_path(tf_record_desc_path, dir_path)
        with file_io.FileIO(tf_record_spec_path, "r") as f:
            spec = json.load(f)
        assert spec["version"] == 1, "TFRecordSpec parsing error: Unsupported version."

        # features types
        type_map = {
            "FloatList": tf.float32,
            "Int64List": tf.int64,
            "BytesList": tf.int8
        }
        feature_info = [FeatureInfo(fi["name"], type_map[fi["kind"]], fi["tags"])
                        for fi in spec["features"]]
        assert len(feature_info) > 0, "TFRecordSpec parsing error: No feature found."

        # groups by multispec
        multispec_feature_groups = []  # type: List[List[str]]
        if "multispec-id" in feature_info[0][2]:
            d = OrderedDict()  # type: OrderedDict[int, List[str]]
            for name, _, tags in feature_info:
                key = int(tags["multispec-id"])
                if key not in d:
                    d[key] = []
                d[key].append(name)
            multispec_feature_groups = [[str()]] * len(d)
            for i, f in six.iteritems(d):
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
    def __get_tf_record_spec_path(tf_record_desc_path,  # type: str
                                  dir_path  # type: str
                                  ):
        # type: (...) -> Text
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
