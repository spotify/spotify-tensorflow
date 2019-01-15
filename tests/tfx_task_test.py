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

from unittest import TestCase

import luigi
from spotify_tensorflow.luigi.tfx_task import TFXBaseTask, TFTransformTask
from tests.test_utils import MockGCSTarget


class DummyRawFeature(luigi.ExternalTask):
    def output(self):
        return MockGCSTarget("output_uri")


class DummyUserTfxTask(TFXBaseTask):
    project = "dummy"
    staging_location = "staging_uri"
    python_script = "mytfx.py"
    requirements_file = "tfx_requirement.txt"

    def requires(self):
        return {"input": DummyRawFeature()}

    def args(self):
        return ["--foo=bar"]

    def output(self):
        return MockGCSTarget(path="output_uri")


class TFXBaseTaskTest(TestCase):
    def test_tfx_task(self):
        task = DummyUserTfxTask()

        expected = [
            "python",
            "mytfx.py",
            "--runner=DataflowRunner",
            "--project=dummy",
            "--input=output_uri/part-*",
            "--output=output_uri",
            "--staging_location=staging_uri",
            "--requirements_file=tfx_requirement.txt",
            "--job_name=dummyusertfxtask",
            "--foo=bar"
        ]
        actual = task._mk_cmd_line()
        self.assertEquals(actual[:2], expected[:2])
        self.assertEquals(set(actual[2:]), set(expected[2:]))


class NoSchemaTftTask(TFTransformTask):
    project = "dummy"
    staging_location = "staging_uri"
    python_script = "mytft.py"
    requirements_file = "tft_requirement.txt"
    job_name = "dummyusertfttask-test"

    def requires(self):
        return {"input": DummyRawFeature()}

    def args(self):
        return ["--foo=bar"]

    def output(self):
        return MockGCSTarget(path="output_uri")


class DummyUserTftTask(NoSchemaTftTask):
    def get_schema_file(self):
        return "schema.pbtxt"


class TFTransformTaskTest(TestCase):
    def test_tft_task(self):
        task = DummyUserTftTask()

        expected = [
            "python",
            "mytft.py",
            "--runner=DataflowRunner",
            "--project=dummy",
            "--input=output_uri/part-*",
            "--output=output_uri",
            "--staging_location=staging_uri",
            "--requirements_file=tft_requirement.txt",
            "--schema_file=schema.pbtxt",
            "--job_name=dummyusertfttask-test",
            "--foo=bar"
        ]
        actual = task._mk_cmd_line()
        self.assertEquals(actual[:2], expected[:2])
        self.assertEquals(set(actual[2:]), set(expected[2:]))

    def test_no_schema_defined_task(self):
        try:
            NoSchemaTftTask()
            self.assertTrue(False)
        except TypeError:
            self.assertTrue(True)
