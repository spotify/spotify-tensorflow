# -*- coding: utf-8 -*-
#
#  Copyright 2017 Spotify AB.
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

from tensorflow.python.lib.io import file_io
from tensorflow.contrib.data import Dataset
from tensorflow.contrib.data import TFRecordDataset

import multiprocessing as mp
import tensorflow as tf

from collections import namedtuple
DatasetContext = namedtuple('DatasetContext', ['filenames_placeholder', 'num_features'])


class Datasets(object):
    _default_feature_desc_filename = '_feature_desc'

    @staticmethod
    def get_parse_proto_function(feature_desc_path, gen_spec):
        """
        Get a function to parse Example proto using given features description.
        Parse function takes a single argument a proto object.

        :param feature_desc_path: filepath to feature description file
        :type feature_desc_path: String

        :return: a tuple with 2 elements: (number of features, parse function)

        :Example:

        TODO
        """
        assert isinstance(feature_desc_path, str), \
            "dir_path is not a String: %r" % feature_desc_path
        assert file_io.file_exists(feature_desc_path), \
            "feature desc `%s` does not exist" % feature_desc_path
        assert isinstance(gen_spec, list), "gen_spec has to be a list: %r" % gen_spec

        def get_features(fpath):
            features = {}
            with file_io.FileIO(fpath, 'r') as f:
                for l in f.readlines():
                    features[l.strip()] = tf.FixedLenFeature((), tf.float32, default_value=0.0)
            return features

        feature_spec = get_features(feature_desc_path)

        def _parse_function(example_proto):
            parsed_features = tf.parse_single_example(example_proto, feature_spec)
            r = tuple(parsed_features.pop(i) for i in gen_spec)
            return r, tuple(parsed_features.values())
        features_len = len(feature_spec) - len(gen_spec)
        return features_len, _parse_function

    @staticmethod
    def get_featran_example_dataset(dir_path,
                                    feature_desc_path=None,
                                    num_threads=mp.cpu_count(),
                                    num_threads_per_file=1,
                                    out_buffer_len=100000,
                                    block_length=10,
                                    gen_spec=[],
                                    compression_type=None):
        """
        Get Dataset of parsed Example protos.

        :param dir_path: directory path contains features
        :type dir_path: String
        :param feature_desc_path: filepath to feature description file
        :type feature_desc_path: String
        :param compression_type: A `tf.string` scalar evaluating to one of `""` (no compression)
                                 `"ZLIB"`, or `"GZIP"`
        :type compression_type: tf.string

        :return: dataset, dict

        :Example:

        TODO
        """
        assert isinstance(dir_path, str), "dir_path is not a String: %r" % dir_path
        assert isinstance(feature_desc_path, str) or feature_desc_path is None, \
            "dir_path is not a String: %r" % feature_desc_path
        assert file_io.file_exists(dir_path), "directory `%s` does not exist" % dir_path
        assert file_io.is_directory(dir_path), "`%s` is not a directory" % dir_path
        if feature_desc_path:
            assert file_io.file_exists(feature_desc_path), \
                    "feature desc `%s` does not exist" % feature_desc_path

        from os.path import join as pjoin
        flist = file_io.list_directory(dir_path)
        input_files = [pjoin(dir_path, x) for x in filter(lambda x: not x.startswith('_'), flist)]
        if feature_desc_path is None:
            feature_desc_path = pjoin(dir_path, Datasets._default_feature_desc_filename)
        filenames = tf.placeholder_with_default(input_files, shape=[None])
        num_features, parse_fn = Datasets.get_parse_proto_function(feature_desc_path, gen_spec)
        dataset = (Dataset.from_tensor_slices(filenames)
                   .interleave(lambda x: TFRecordDataset(x, compression_type)
                               .map(parse_fn,
                                    num_threads=num_threads_per_file,
                                    output_buffer_size=out_buffer_len),
                               cycle_length=num_threads, block_length=block_length))
        return dataset, DatasetContext(filenames_placeholder=filenames, num_features=num_features)
