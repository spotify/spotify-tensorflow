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

from unittest import TestCase

import luigi
from luigi.contrib.gcs import GCSTarget
from spotify_tensorflow.luigi.tfx_task import TFXBaseTask


class DummyRawFeature(luigi.ExternalTask):
    def output(self):
        return GCSTarget("output_uri")


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
        return GCSTarget(path="output_uri")


class TFXBaseTaskTest(TestCase):

    def test_task(self):
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
        expected.sort()
        actual = task._mk_cmd_line()
        actual.sort()
        self.assertEquals(actual, expected)
