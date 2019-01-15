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
from spotify_tensorflow.luigi.python_dataflow_task import PythonDataflowTask
from tests.test_utils import MockGCSTarget


class DummyRawFeature(luigi.ExternalTask):
    def output(self):
        return MockGCSTarget("output_uri")


class DummyPythonDataflowTask(PythonDataflowTask):
    python_script = "pybeamjob.py"
    requirements_file = "tfx_requirement.txt"
    zone = "zone"
    region = "region"
    project = "dummy"
    worker_machine_type = "n1-standard-4"
    num_workers = 5
    max_num_workers = 20
    autoscaling_algorithm = "THROUGHPUT_BASED"
    service_account = "dummy@dummy.iam.gserviceaccount.com"
    local_runner = True
    staging_location = "staging_uri"
    temp_location = "tmp"
    network = "network"
    subnetwork = "subnetwork"
    disk_size_gb = 30
    worker_disk_type = "disc_type"
    job_name = "dummy"
    setup_file = "setup.py"

    def requires(self):
        return {"input": DummyRawFeature()}

    def args(self):
        return ["--foo=bar"]

    def output(self):
        return MockGCSTarget(path="output_uri")


class PythonDataflowTaskFailedOnValidation(PythonDataflowTask):
    python_script = "pybeamjob.py"

    # override to construct a test run
    def _mk_cmd_line(self):
        return ["python", "-c", "\"print(1)\""]

    def validate_output(self):
        return False

    def args(self):
        return ["--foo=bar"]

    def output(self):
        return MockGCSTarget(path="output_uri")


class PythonDataflowTaskTest(TestCase):
    def test_python_dataflow_task(self):
        task = DummyPythonDataflowTask()

        expected = [
            "python",
            "pybeamjob.py",
            "--runner=DirectRunner",
            "--project=dummy",
            "--autoscaling_algorithm=THROUGHPUT_BASED",
            "--num_workers=5",
            "--max_num_workers=20",
            "--service_account_email=dummy@dummy.iam.gserviceaccount.com",
            "--input=output_uri/part-*",
            "--output=output_uri",
            "--staging_location=staging_uri",
            "--requirements_file=tfx_requirement.txt",
            "--worker_machine_type=n1-standard-4",
            "--foo=bar",
            "--temp_location=tmp",
            "--network=network",
            "--subnetwork=subnetwork",
            "--disk_size_gb=30",
            "--worker_disk_type=disc_type",
            "--job_name=dummy",
            "--zone=zone",
            "--region=region",
            "--setup_file=setup.py"
        ]
        actual = task._mk_cmd_line()
        self.assertEquals(actual[:2], expected[:2])
        self.assertEquals(set(actual[2:]), set(expected[2:]))

    def test_task_failed_on_validation(self):
        task = PythonDataflowTaskFailedOnValidation()
        try:
            task.run()
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)
