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

import os
import tempfile
from os.path import join as pjoin

import tensorflow as tf
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.lib.io.tf_record import TFRecordWriter
from spotify_tensorflow.dataset import Datasets
from spotify_tensorflow.trainer import Trainer


class DataUtil(object):
    tf_record_spec = ("""{"version":1,""" +
                      """"features":[{"name":"f1","kind":"Int64List","tags":{}},""" +
                      """{"name":"f2","kind":"Int64List","tags":{}}],""" +
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
    def test_get_featran_example_dataset(self):
        d, _, _ = DataUtil.write_featran_test_data()
        with self.test_session() as sess:
            dataset, c = Datasets._get_featran_example_dataset(d)
            self.assertEquals(len(c.features), 2)
            iterator = dataset.make_one_shot_iterator()
            r = iterator.get_next()
            f1, f2 = r["f1"], r["f2"]
            self.assertAllEqual([1, 2], sess.run([f1, f2]))
            with self.assertRaises(tf.errors.OutOfRangeError):
                f1.eval()

    test_resources_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                      "resources/tf-test-resource/tf-records/train")
    N_FEATURES = 5
    N_Y = 1
    N_X = N_FEATURES - N_Y
    N_POINTS = 1817

    def test_mk_iter(self):
        it, context = Datasets.mk_iter(self.test_resources_dir)
        batch_it = it.get_next()

        with tf.Session() as sess:
            batch = sess.run(batch_it)
            self.assertEqual(len(batch), self.N_FEATURES)

            first_feature = list(context.features.keys())[0]
            self.assertEqual(len(batch[first_feature]), tf.flags.FLAGS["batch-size"].value)

    def test_data_frame_read_dataset(self):
        data = Datasets.dataframe.read_dataset(self.test_resources_dir)
        self.assertEqual(len(data), self.N_POINTS)

    def test_data_frame_read_dataset_dictonary(self):
        data = Datasets.dict.read_dataset(self.test_resources_dir)
        self.assertEqual(len(data), self.N_FEATURES)
        self.assertEqual(len(data["f1"]), self.N_POINTS)

    def test_data_frame_batch_iterator(self):
        batch_size = 10
        it = Datasets.dataframe.batch_iterator(self.test_resources_dir, batch_size)
        batches = [df for df in it]
        total = 0
        for df in batches[:-1]:
            n, f = df.shape
            self.assertEqual(n, batch_size)
            self.assertEqual(f, self.N_FEATURES)
            total += n
        last_batch_len = len(batches[-1])
        self.assertLessEqual(last_batch_len, batch_size)
        self.assertEqual(total + last_batch_len, self.N_POINTS)

    def test_data_frame_unpack_multispec(self):
        # dataset was saved using multispec: `val dataset = MultiFeatureSpec(features, label)`
        X, Y = Datasets.dataframe.read_dataset(self.test_resources_dir, unpack_multispec=True)
        n_X, f_X = X.shape
        self.assertEqual(n_X, self.N_POINTS)
        self.assertEqual(f_X, self.N_X)
        n_Y, f_Y = Y.shape
        self.assertEqual(n_Y, self.N_POINTS)
        self.assertEqual(f_Y, self.N_Y)

    def test_feature_order_multispec_dataframe(self):
        expected_features = ["f3", "f1", "f2_EVEN", "f2_ODD"]
        df, _ = Datasets.dataframe.read_dataset(self.test_resources_dir, unpack_multispec=True)
        self.assertEqual(list(df.columns.values), expected_features)

    def test_feature_order_dataframe(self):
        expected_features = ["f3", "f1", "f2_EVEN", "f2_ODD", "label"]
        df = Datasets.dataframe.read_dataset(self.test_resources_dir)
        self.assertEqual(list(df.columns.values), expected_features)

    def test_feature_order_multispec(self):
        expected_features = ["f3", "f1", "f2_EVEN", "f2_ODD"]
        _, context = Datasets.mk_iter(self.test_resources_dir)
        feature_names, _ = context.multispec_feature_groups
        self.assertEqual(feature_names, expected_features)

    def test_trainer_shouldnt_crash(self):
        context = Datasets.get_context(self.test_resources_dir)
        (feature_names, label_names) = context.multispec_feature_groups
        feature_columns = [tf.feature_column.numeric_column(name) for name in feature_names]

        config = Trainer.get_default_run_config(job_dir=tempfile.mkdtemp())

        estimator = tf.estimator.LinearClassifier(feature_columns=feature_columns,
                                                  config=config)

        def split_features_label_fn(parsed_features):
            self.assertEqual(len(label_names),  1)
            label = parsed_features.pop(label_names[0])
            return parsed_features, label

        Trainer.run(estimator,
                    training_data_dir=self.test_resources_dir,
                    eval_data_dir=self.test_resources_dir,
                    split_features_label_fn=split_features_label_fn,
                    run_config=config)
