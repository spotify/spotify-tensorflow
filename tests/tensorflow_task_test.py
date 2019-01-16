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

import os
import tempfile
from unittest import TestCase
from subprocess import CalledProcessError

import luigi
from luigi import LocalTarget
from spotify_tensorflow.luigi.tensorflow_task import TensorFlowTask
from tests.test_utils import MockGCSTarget


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
            "gcloud", "ml-engine", "local", "train",
            "--package-path=/path/to/package",
            "--module-name=models.my_tf_model",
            "--",
            "--training-data=gs://training/data",
            "--job-dir=/local/job/dir",
            "--arg1=foo",
            "--arg2=bar"
        ]
        self.assertEquals(task._mk_cmd(), expected)

    def test_cloud_task(self):
        task = DummyTask(cloud=True,
                         model_package_path="/path/to/package",
                         job_dir="gs://job/dir",
                         ml_engine_conf="/path/conf.yaml",
                         blocking=True,
                         runtime_version="foo",
                         scale_tier="bar",
                         tf_debug=True)
        expected = [
            "gcloud", "ml-engine",
            "--project=project-1",
            "jobs", "submit", "training",
            ".*_DummyTask_.*",
            "--region=europe-west1",
            "--config=/path/conf.yaml",
            "--job-dir=gs://job/dir",
            "--stream-logs",
            "--runtime-version=foo",
            "--scale-tier=bar",
            "--package-path=/path/to/package",
            "--module-name=models.my_tf_model",
            "--verbosity=debug",
            "--",
            "--training-data=gs://training/data",
            "--arg1=foo",
            "--arg2=bar"
        ]
        actual = task._mk_cmd()
        self.assertEquals(actual[:6], expected[:6])
        self.assertRegexpMatches(actual[6], expected[6])
        self.assertEquals(actual[7:], expected[7:])

    def test_run_success(self):

        tmp_dir_path = tempfile.mkdtemp()

        class SuperDummyTask(DummyTask):

            def _mk_cmd(self):
                return ["echo", "hello"]

            def requires(self):
                return LocalTarget(tmp_dir_path)

        task = SuperDummyTask(cloud=False,
                              job_dir=tmp_dir_path,
                              model_package_path="/path/to/package")
        task.run()

        assert os.path.exists(os.path.join(tmp_dir_path, "_SUCCESS"))

    def test_run_fail(self):

        tmp_dir_path = tempfile.mkdtemp()

        class SuperDummyTask(DummyTask):

            def _mk_cmd(self):
                return ["cp", "some", "crap"]

            def requires(self):
                return LocalTarget(tmp_dir_path)

        task = SuperDummyTask(cloud=False,
                              job_dir=tmp_dir_path,
                              model_package_path="/path/to/package")

        try:
            task.run()
            assert False
        except CalledProcessError:
            assert not os.path.exists(os.path.join(tmp_dir_path, "_SUCCESS"))
