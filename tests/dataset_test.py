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

from os.path import join as pjoin
from spotify_tensorflow.dataset import Datasets
import tensorflow as tf

from tensorflow.python.lib.io.tf_record import TFRecordWriter
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2


class DataUtil(object):

    @staticmethod
    def write_featran_test_data(feature_desc=['f1', 'f2'],
                                values=[{'f1': 1., 'f2': 2.}],
                                feature_desc_filename='_feature_desc'):
        tmp_dir = tf.test.get_temp_dir()
        feature_desc_file = pjoin(tmp_dir, feature_desc_filename)
        with open(feature_desc_file, 'w') as f:
            f.writelines('\n'.join(feature_desc))
        e = DataUtil.get_example_proto(values)
        data_file = pjoin(tmp_dir, 'test.tfrecord')
        with TFRecordWriter(data_file) as f:
            for i in e:
                f.write(i.SerializeToString())
        return tmp_dir, data_file, feature_desc_file

    @staticmethod
    def get_example_proto(values=[{'f1': 1., 'f2': 2.}]):
        return [example_pb2.Example(features=feature_pb2.Features(feature={
                    k: feature_pb2.Feature(float_list=feature_pb2.FloatList(value=[v]))
                    for k, v in d.items()
                })) for d in values]


class SquareTest(tf.test.TestCase):

    def testGetFeatranExampleDataset(self):
        d, _, _ = DataUtil.write_featran_test_data()
        with self.test_session() as sess:
            dataset, c = Datasets.get_featran_example_dataset(d)
            self.assertEquals(list(c.feature_names), ['f1', 'f2'])
            self.assertEquals(c.num_features, 2)
            iterator = dataset.make_one_shot_iterator()
            r = iterator.get_next()
            f1, f2 = r['f1'], r['f2']
            self.assertAllEqual([1., 2.], sess.run([f1, f2]))
            with self.assertRaises(tf.errors.OutOfRangeError):
                f1.eval()
