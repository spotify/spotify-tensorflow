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

import json
import os
import shutil
from tempfile import mkdtemp, NamedTemporaryFile
from typing import Dict, List  # noqa: F401
from unittest import TestCase

import tensorflow as tf
import tensorflow_transform as tft
from spotify_tensorflow.example_decoders import ExampleDecoder
from spotify_tensorflow.tf_schema_utils import schema_txt_to_feature_spec
from spotify_tensorflow.tfx.tft import TFTransform


def dummy_preprocessing_fn(inputs):
    out = dict()
    out["test_feature_fx"] = tft.scale_to_z_score(inputs["test_feature"])
    return out


class TFTransformTest(TestCase):
    schema_txt = """
            feature {
                name: "test_feature"
                value_count {
                    min: 1
                    max: 1
                }
                type: FLOAT
                presence {
                    min_count: 1
                }
            }
        """

    def setUp(self):
        self.schema_file = NamedTemporaryFile(delete=False)
        self.schema_file_name = self.schema_file.name
        self.schema_file.write(self.schema_txt)
        self.schema_file.close()
        self.feature_spec = schema_txt_to_feature_spec(self.schema_file_name)

        self.train_data = NamedTemporaryFile(suffix=".tfrecords", delete=False)
        self.train_data_file = self.train_data.name
        self.train_data.close()
        train_dicts = [{"test_feature": [0.1]}, {"test_feature": [-0.1]}]
        with tf.python_io.TFRecordWriter(self.train_data_file) as writer:
            for train_dict in train_dicts:
                tf_example = build_tf_example_from_dict(train_dict)
                writer.write(tf_example.SerializeToString())

        self.eval_data = NamedTemporaryFile(suffix=".tfrecords", delete=False)
        self.eval_data_file = self.eval_data.name
        self.eval_data.close()
        eval_dicts = [{"test_feature": [0.2]}, {"test_feature": [-0.2]}]
        with tf.python_io.TFRecordWriter(self.eval_data_file) as writer:
            for eval_dict in eval_dicts:
                tf_example = build_tf_example_from_dict(eval_dict)
                writer.write(tf_example.SerializeToString())

        self.output_dir = mkdtemp()
        self.temp_dir = mkdtemp()

    def test_transform(self):
        pipeline_args = ["--runner=DirectRunner"]
        tft_args = ["--training_data=%s" % self.train_data_file,
                    "--evaluation_data=%s" % self.eval_data_file,
                    "--output_dir=%s" % self.output_dir,
                    "--temp_location=%s" % self.temp_dir,
                    "--schema_file=%s" % self.schema_file_name]
        args = tft_args + pipeline_args
        TFTransform(preprocessing_fn=dummy_preprocessing_fn).run(args=args)

        # test output structure
        sub_folders = os.listdir(self.output_dir)
        self.assertEquals(set(sub_folders),
                          {"evaluation", "training", "transform_fn", "transformed_metadata"})
        transformed_train_files = [f for f in os.listdir(os.path.join(self.output_dir, "training"))
                                   if f.endswith(".tfrecords")]
        self.assertEquals(len(transformed_train_files), 1)
        transformed_eval_files = [f for f in os.listdir(os.path.join(self.output_dir, "evaluation"))
                                  if f.endswith(".tfrecords")]
        self.assertEquals(len(transformed_eval_files), 1)
        transform_fn_file = os.path.join(self.output_dir, "transform_fn", "saved_model.pb")
        self.assertTrue(os.path.exists(transform_fn_file))

        # test transformed training data
        path = os.path.join(self.output_dir, "training", transformed_train_files[0])
        transformed_train = [js["features"]["feature"]["test_feature_fx"]["floatList"]["value"]
                             for js in parse_tf_records(path)]
        transformed_train.sort(key=lambda x: x[0])
        self.assertEqual(len(transformed_train), 2)
        self.assertEqual(transformed_train, [[-1.0], [1.0]])

        # test transformed evaluation data
        path = os.path.join(self.output_dir, "evaluation", transformed_eval_files[0])
        transformed_eval = [js["features"]["feature"]["test_feature_fx"]["floatList"]["value"]
                            for js in parse_tf_records(path)]
        transformed_eval.sort(key=lambda x: x[0])
        self.assertEqual(len(transformed_eval), 2)
        # transformed_eval is derived from the z-score transformation based on the training data
        # eval_value = (raw_value - train_mean) / train_std_dev
        self.assertEqual(transformed_eval, [[-2.0], [2.0]])

    def test_no_train_no_transform_fn_dir(self):
        pipeline_args = ["--runner=DirectRunner"]
        tft_args = ["--evaluation_data=%s" % self.eval_data_file,
                    "--output_dir=%s" % self.output_dir,
                    "--temp_location=%s" % self.temp_dir,
                    "--schema_file=%s" % self.schema_file_name]
        args = tft_args + pipeline_args
        try:
            TFTransform(preprocessing_fn=dummy_preprocessing_fn).run(args=args)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def tearDown(self):
        os.remove(self.train_data_file)
        os.remove(self.eval_data_file)
        os.remove(self.schema_file.name)
        shutil.rmtree(self.output_dir)
        shutil.rmtree(self.temp_dir)


def build_tf_example_from_dict(dict_data):
    # type: (Dict[str, list]) -> tf.train.Example
    feature_dict = {}
    for feature_name in dict_data.keys():
        feature_value = dict_data[feature_name]
        tf_feature = tf.train.Feature(float_list=tf.train.FloatList(value=feature_value))
        feature_dict[feature_name] = tf_feature
    return tf.train.Example(features=tf.train.Features(feature=feature_dict))


def parse_tf_records(tf_records_file):
    # type: (str) -> List[dict]
    record_iterator = tf.python_io.tf_record_iterator(path=tf_records_file)
    data = []
    example_decoder = ExampleDecoder()
    for string_record in record_iterator:
        json_str = example_decoder.to_json(string_record)
        data.append(json.loads(json_str))
    return data
