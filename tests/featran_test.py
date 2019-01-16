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

import os

import tensorflow as tf
from spotify_tensorflow.featran import Featran


class FeatranTest(tf.test.TestCase):
    test_resources_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources")
    settings_dir = os.path.join(test_resources_dir, "tf-test-resource/tf-records/settings")

    def test_settings(self):
        settings = Featran.settings(self.settings_dir)
        self.assertEqual(len(settings), 4)
        self.assertEqual(settings[0]["name"], "f3")

    def test_name(self):
        names = Featran.names(self.settings_dir)
        self.assertEqual(len(names), 5)

    def test_splits(self):
        def feature_splitter(setting):
            return "labels" if setting == "label" else "features"

        names = Featran.names(self.settings_dir, feature_splitter)
        self.assertEqual(len(names), 2)
        self.assertEqual(len(names["features"]), 4)
        self.assertEqual(len(names["labels"]), 1)
