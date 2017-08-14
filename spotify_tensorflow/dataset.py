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
from tensorflow.contrib.data import TFRecordDataset

import tensorflow as tf

class Datasets(object):

    @staticmethod
    def get_parse_proto_function(features_filepath, gen_spec):
        def get_features(fpath):
            features = {}
            with file_io.FileIO(fpath, 'r') as f:
                for l in f.readlines():
                    features[l.strip()] = tf.FixedLenFeature((), tf.float32)
            return features
    
        feature_spec = get_features(features_filepath)
        def _parse_function(example_proto):
            parsed_features = tf.parse_single_example(example_proto, feature_spec)
            r = tuple(parsed_features.pop(i) for i in gen_spec)
            return r, tuple(parsed_features.values())
        return _parse_function

    @staticmethod
    def get_example_dataset(feature_info_filepath, gen_spec=None):
        """
        Get Dataset of parsed Example protos.

        :param feature_info_filepath: filepath of feature info file
        :type feature_info_filepath: String

        :return: filename-placeholder, dataset

        :Example:

        TODO
        """
        filenames = tf.placeholder(tf.string, shape=[None])
        dataset = TFRecordDataset(filenames)
        dataset = dataset.map(Datasets.get_parse_proto_function(feature_info_filepath, gen_spec))
        return filenames, dataset

