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
#

import os
import tempfile
from unittest import TestCase

from examples.examples_utils import get_taxi_data_dir
from spotify_tensorflow.tfx.tfdv import GenerateStats


class TFDVTest(TestCase):
    def setUp(self):
        taxi_data = get_taxi_data_dir()
        self.input_data_dir = taxi_data
        self.stats_file = os.path.join(taxi_data, "_stats.pb")
        tmp_dir = tempfile.mkdtemp()
        self.pipeline_args = [
            "--temp_location=%s" % tmp_dir,
            "--staging_location=%s" % tmp_dir,
            "--runner=DirectRunner"
        ]

    def test_generate_stats(self):
        GenerateStats(self.input_data_dir).run(self.pipeline_args)
        self.assertTrue(os.path.exists(self.stats_file))

    def tearDown(self):
        os.remove(self.stats_file)
