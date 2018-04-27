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
from spotify_tensorflow.luigi.tensorflow_task import TensorFlowTask


class MockGCSTarget(GCSTarget):
    def __init__(self, path):
        self.path = path


class TestRequires(luigi.ExternalTask):
    def output(self):
        return MockGCSTarget("gs://training/data")


class DummyTask(TensorFlowTask):
    model_package = "models"
    model_name = "my_tf_model"
    gcp_project = "project-1"
    region = "europe-west1"
    model_name_suffix = "tf_model1"

    def tf_task_args(self):
        return ["--arg1=foo", "--arg2=bar"]

    def requires(self):
        return {"training-data": TestRequires()}


class TensorflowTaskTest(TestCase):

    def test_local_task(self):
        task = DummyTask(cloud=False,
                         job_dir="/local/job/dir",
                         model_package_path="/path/to/package")
        expected = [
            "gcloud ml-engine local train",
            "--package-path=/path/to/package",
            "--module-name=models.my_tf_model",
            "--",
            "--training-data=gs://training/data",
            "--job_dir=/local/job/dir",
            "--arg1=foo",
            "--arg2=bar"
        ]
        self.assertEquals(task._mk_cmd(), " ".join(expected))

    def test_cloud_task(self):
        task = DummyTask(cloud=True,
                         model_package_path="/path/to/package",
                         job_dir="gs://job/dir",
                         ml_engine_conf="/path/conf.yaml",
                         blocking=False,
                         tf_debug=True)
        expected = [
            "gcloud", "ml-engine",
            "--project=project-1",
            "jobs", "submit", "training",
            ".*_DummyTask_.*",
            "--region=europe-west1",
            "--config=/path/conf.yaml",
            "--job_dir=gs://job/dir",
            "--package-path=/path/to/package",
            "--module-name=models.my_tf_model",
            "--verbosity=debug",
            "--",
            "--training-data=gs://training/data",
            "--arg1=foo",
            "--arg2=bar"
        ]
        actual = task._mk_cmd().split(" ")
        self.assertEquals(actual[:6], expected[:6])
        self.assertRegexpMatches(actual[6], expected[6])
        self.assertEquals(actual[7:], expected[7:])
