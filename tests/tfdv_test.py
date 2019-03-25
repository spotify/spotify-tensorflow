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
from os.path import join as pjoin
from unittest import TestCase
import mock

import apache_beam as beam
from examples.examples_utils import get_taxi_data_dir
from spotify_tensorflow.tfx.tfdv import TfDataValidator, generate_statistics_from_bq


class TfDataValidatorTest(TestCase):
    def setUp(self):
        taxi_data = get_taxi_data_dir()
        self.data_location = taxi_data
        self.schema = pjoin(taxi_data, "chicago_taxi_schema.pbtxt")
        self.schema_snapshot_path = pjoin(taxi_data, "schema_snapshot.pb")
        self.stats_file = pjoin(taxi_data, "stats.pb")
        self.anomalies_path = pjoin(self.data_location, "anomalies.pb")
        tmp_dir = tempfile.mkdtemp()
        self.pipeline_args = [
            "--temp_location=%s" % tmp_dir,
            "--staging_location=%s" % tmp_dir,
            "--runner=DirectRunner"
        ]

    def test_write_stats_and_schema(self):
        TfDataValidator(self.schema, self.data_location).write_stats_and_schema(self.pipeline_args)
        self.assertTrue(os.path.exists(self.stats_file))
        self.assertTrue(os.path.exists(self.schema_snapshot_path))

    def test_validate_stats_against_schema(self):
        validator = TfDataValidator(self.schema, self.data_location)
        validator.write_stats(self.pipeline_args)
        self.assertTrue(os.path.exists(self.stats_file))
        has_no_anomalies = validator.validate_stats_against_schema()
        self.assertFalse(has_no_anomalies)
        self.assertTrue(os.path.exists(self.anomalies_path))

    @mock.patch("apache_beam.io.Read", return_value=beam.Create())
    def test_generate_statistics_from_bq(self, Read):
        # table_rows = [
        #     bigquery.TableRow(f=[]),
        #     bigquery.TableRow(f=[])
        # ]
        #
        # schema = bigquery.TableSchema(
        #     fields=[
        #         bigquery.TableFieldSchema(name='b', type='BOOLEAN', mode='REQUIRED'),
        #         bigquery.TableFieldSchema(name='b', type='BOOLEAN', mode='REQUIRED')
        #     ]
        # )
        #
        # expected_rows = [{"a": "1", "b": [1, 2]}, {{"a": "1", "b": [3, 4]}}]

        generate_statistics_from_bq(self.pipeline_args, "SELECT * FROM TABLE", self.stats_file)

    def tearDown(self):
        if os.path.exists(self.stats_file):
            os.remove(self.stats_file)
        if os.path.exists(self.schema_snapshot_path):
            os.remove(self.schema_snapshot_path)
        if os.path.exists(self.anomalies_path):
            os.remove(self.anomalies_path)
