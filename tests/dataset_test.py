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

from functools import wraps
import os
import tempfile
from os.path import join as pjoin

import tensorflow as tf
from tensorflow.python.platform import test
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.lib.io.tf_record import TFRecordWriter
from tensorflow.python.eager.context import eager_mode
from spotify_tensorflow.dataset import Datasets
from spotify_tensorflow.featran import Featran
from spotify_tensorflow.tf_schema_utils import feature_spec_to_schema


class DataUtil(object):

    @staticmethod
    def write_test_data(example_proto,
                        schema,
                        schema_filename="schema.pb"):
        tmp_dir = tf.test.get_temp_dir()
        schema_path = pjoin(tmp_dir, schema_filename)
        with open(schema_path, "wb") as f:
            f.write(schema.SerializeToString())
        data_file = pjoin(tmp_dir, "test.tfrecord")
        with TFRecordWriter(data_file) as f:
            for i in example_proto:
                f.write(i.SerializeToString())
        return data_file, schema_path

    @staticmethod
    def run_in_eager(f):
        @wraps(f)
        def wrapper(*args, **kwds):
            with eager_mode():
                return f(*args, **kwds)
        return wrapper


class SparseTest(test.TestCase):

    @staticmethod
    def _write_test_data():
        schema = feature_spec_to_schema({"f0": tf.VarLenFeature(dtype=tf.int64),
                                         "f1": tf.VarLenFeature(dtype=tf.int64),
                                         "f2": tf.VarLenFeature(dtype=tf.int64)})
        batches = [
            [1, 4, None],
            [2, None, None],
            [3, 5, None],
            [None, None, None],
        ]

        example_proto = [example_pb2.Example(features=feature_pb2.Features(feature={
            "f" + str(i): feature_pb2.Feature(int64_list=feature_pb2.Int64List(value=[f]))
            for i, f in enumerate(batch) if f is not None
        })) for batch in batches]

        return DataUtil.write_test_data(example_proto, schema)

    @DataUtil.run_in_eager
    def test_sparse_features(self):
        data, schema_path = SparseTest._write_test_data()
        dataset = next(Datasets.dataframe.examples_via_schema(data,
                                                              schema_path,
                                                              shuffle=False))  # noqa: E501
        values = dataset.values
        self.assertSequenceEqual([1, 4, 0], list(values[0]))
        self.assertSequenceEqual([2, 0, 0], list(values[1]))
        self.assertSequenceEqual([3, 5, 0], list(values[2]))
        self.assertSequenceEqual([0, 0, 0], list(values[3]))

    @DataUtil.run_in_eager
    def test_sparse_features_with_default(self):
        data, schema_path = SparseTest._write_test_data()
        d = 1
        dataset = next(Datasets.dataframe.examples_via_schema(data,
                                                              schema_path,
                                                              default_value=d,
                                                              shuffle=False))  # noqa: E501
        values = dataset.values
        self.assertSequenceEqual([1, 4, d], list(values[0]))
        self.assertSequenceEqual([2, d, d], list(values[1]))
        self.assertSequenceEqual([3, 5, d], list(values[2]))
        self.assertSequenceEqual([d, d, d], list(values[3]))


class SquareTest(test.TestCase):

    @staticmethod
    def _write_test_data():
        schema = feature_spec_to_schema({"f1": tf.FixedLenFeature((), tf.int64),
                                         "f2": tf.FixedLenFeature((), tf.int64)})
        values = [{"f1": 1, "f2": 2}]

        example_proto = [example_pb2.Example(features=feature_pb2.Features(feature={
            k: feature_pb2.Feature(int64_list=feature_pb2.Int64List(value=[v]))
            for k, v in d.items()
        })) for d in values]

        return DataUtil.write_test_data(example_proto, schema)

    def test_simple_get_example_dataset(self):
        data, schema_path = SquareTest._write_test_data()
        with self.test_session() as sess:
            dataset = Datasets.examples_via_schema(data, schema_path)  # noqa: E501
            iterator = dataset.make_one_shot_iterator()
            r = iterator.get_next()
            f1, f2 = r["f1"], r["f2"]
            self.assertAllEqual([[1], [2]], sess.run([f1, f2]))
            with self.assertRaises(tf.errors.OutOfRangeError):
                f1.eval()

    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "resources", "tf-test-resource", "tf-records")

    train_data = os.path.join(data_dir, "train", "part-*")
    eval_data = os.path.join(data_dir, "eval", "part-*")
    schema_path = os.path.join(data_dir, "train", "_inferred_schema.pb")
    stats_path = os.path.join(data_dir, "train", "_stats.pb")
    settings_path = os.path.join(data_dir, "settings")

    N_FEATURES = 5
    N_Y = 1
    N_X = N_FEATURES - N_Y
    N_POINTS = 792
    ordered_feature_names = Featran.names(settings_path)

    def test_get_example_dataset(self):
        dataset = Datasets.examples_via_schema(self.train_data, self.schema_path, batch_size=16)
        batch_it = dataset.make_one_shot_iterator().get_next()

        with tf.Session() as sess:
            batch = sess.run(batch_it)
            self.assertEqual(len(batch), self.N_FEATURES)
            self.assertEqual(len(batch["f1"]), 16)

    def test_parse_schema_from_stats(self):
        feature_spec, schema = Datasets.parse_schema_from_stats(self.stats_path)
        self.assertEqual(len(feature_spec), self.N_FEATURES)

    @DataUtil.run_in_eager
    def test_data_frame_read_dataset(self):
        data = next(
            Datasets.dataframe.examples_via_schema(self.train_data,
                                                   batch_size=1024,
                                                   schema_path=self.schema_path))
        self.assertEqual(self.N_POINTS, len(data))
        self.assertEqual(self.N_FEATURES, len(data.columns))

    @DataUtil.run_in_eager
    def test_data_frame_read_dataset_dictionary(self):
        data = next(
            Datasets.dict.examples_via_schema(self.train_data,
                                              batch_size=1024,
                                              schema_path=self.schema_path))
        self.assertEqual(self.N_FEATURES, len(data.keys()))
        self.assertEqual(self.N_POINTS, len(data["f1"]))

    @DataUtil.run_in_eager
    def test_data_frame_read_dataset_ordered(self):
        dataset = Datasets.dataframe.examples_via_schema(self.train_data,
                                                         batch_size=1024,
                                                         schema_path=self.schema_path)
        ordered = Featran.reorder_dataframe_dataset(dataset, self.settings_path)
        data = next(ordered)
        self.assertEqual(self.N_POINTS, len(data))
        self.assertEqual(self.N_FEATURES, len(data.columns))
        self.assertEqual(self.ordered_feature_names, data.columns.values.tolist())

    @DataUtil.run_in_eager
    def test_data_frame_read_dataset_dictionary_settings(self):
        dataset = Datasets.dict.examples_via_schema(self.train_data,
                                                    batch_size=1024,
                                                    schema_path=self.schema_path)
        ordered = Featran.reorder_numpy_dataset(dataset, self.settings_path)
        data = next(ordered)
        self.assertEqual(self.N_FEATURES, len(data.keys()))
        self.assertEqual(self.N_POINTS, len(data["f1"]))
        self.assertEqual(self.ordered_feature_names, list(data.keys()))

    @DataUtil.run_in_eager
    def test_data_frame_batch_iterator(self):
        batch_size = 10
        it = Datasets.dataframe.examples_via_schema(self.train_data,
                                                    self.schema_path,
                                                    batch_size=batch_size)
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

    def test_trainer_shouldnt_crash(self):
        label_key = "label"
        feature_spec, _ = Datasets.parse_schema(self.schema_path)
        all_features = {name: tf.feature_column.numeric_column(name, default_value=.0)
                        for name in feature_spec.keys()}
        feature_columns = all_features.copy()
        feature_columns.pop(label_key)

        config = tf.estimator.RunConfig(tempfile.mkdtemp())

        estimator = tf.estimator.LinearClassifier(feature_columns=feature_columns.values(),
                                                  config=config)

        def split_features_label_fn(parsed_features):
            label = parsed_features.pop(label_key)
            return parsed_features, label

        def get_in_fn(data):
            raw_feature_spec = tf.feature_column.make_parse_example_spec(all_features.values())

            def in_fn():
                dataset = Datasets.examples_via_feature_spec(data, raw_feature_spec)
                return dataset.map(split_features_label_fn)

            return in_fn

        estimator.train(get_in_fn(self.train_data)).evaluate(get_in_fn(self.eval_data))
