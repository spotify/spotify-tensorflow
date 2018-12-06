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

from spotify_tensorflow.luigi.beam_base_task import PythonBeamBaseTask


class DummyBeamTask(PythonBeamBaseTask):
    python_script = "pybeamjob.py"

    def extra_cmd_line_args(self):
        cmd_line = ["--foo=bar"]
        return cmd_line


class PythonBeamBaseTaskTest(TestCase):

    def test_task(self):
        task = DummyBeamTask(runner="DataflowRunner",
                             project="dummy",
                             machineWorkerType="n1-standard-4",
                             region="europe-west1",
                             temp_location="tmp")
        expected = [
            "python pybeamjob.py",
            "--runner=DataflowRunner",
            "--project=dummy",
            "--temp_location=tmp",
            "--machineWorkerType=n1-standard-4",
            "--region=europe-west1",
            "--foo=bar"
        ]
        self.assertEquals(task._make_cmd_line(), expected)
