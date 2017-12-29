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

from __future__ import absolute_import, division, print_function

from os.path import join as pjoin

import tensorflow as tf
from spotify_tensorflow.dataset import Datasets
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.lib.io.tf_record import TFRecordWriter


class DataUtil(object):
    tf_record_spec = ("""{"version":1,""" +
                      """"features":[["f1","Int64List",{}],["f2","Int64List",{}]],""" +
                      """"compression":"UNCOMPRESSED"}""")
    values = [{"f1": 1, "f2": 2}]
    tf_record_spec_filename = "_tf_record_spec.json"

    @classmethod
    def write_featran_test_data(cls):
        tmp_dir = tf.test.get_temp_dir()
        feature_desc_file = pjoin(tmp_dir, cls.tf_record_spec_filename)
        with open(feature_desc_file, "w") as f:
            f.write(cls.tf_record_spec)
        e = DataUtil.get_example_proto()
        data_file = pjoin(tmp_dir, "test.tfrecord")
        with TFRecordWriter(data_file) as f:
            for i in e:
                f.write(i.SerializeToString())
        return tmp_dir, data_file, feature_desc_file

    @classmethod
    def get_example_proto(cls):
        return [example_pb2.Example(features=feature_pb2.Features(feature={
            k: feature_pb2.Feature(int64_list=feature_pb2.Int64List(value=[v]))
            for k, v in d.items()
        })) for d in cls.values]


class SquareTest(tf.test.TestCase):
    def testGetFeatranExampleDataset(self):
        d, _, _ = DataUtil.write_featran_test_data()
        with self.test_session() as sess:
            dataset, c = Datasets.get_featran_example_dataset(d)
            self.assertEquals(len(c.feature_names), 2)
            iterator = dataset.make_one_shot_iterator()
            r = iterator.get_next()
            f1, f2 = r["f1"], r["f2"]
            self.assertAllEqual([1, 2], sess.run([f1, f2]))
            with self.assertRaises(tf.errors.OutOfRangeError):
                f1.eval()
