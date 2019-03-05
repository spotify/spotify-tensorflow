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

from unittest import TestCase

import luigi
from spotify_tensorflow.luigi import predict
from spotify_tensorflow.luigi.batch_predict_task import BatchPredictTask


class DummyInput(luigi.ExternalTask):
    def output(self):
        return luigi.LocalTarget("/data/")


class DummyModel(luigi.ExternalTask):
    def output(self):
        return luigi.LocalTarget("/model/")


class DummyBatchPredictTask(BatchPredictTask):
    project = "project"
    temp_location = "temp"
    job_name = "dummy-batch-predict"

    def get_input_dir(self):
        return self.input()["input-data"].path

    def get_model_dir(self):
        return self.input()["model-path"].path

    def requires(self):
        return {
            "input-data": DummyInput(),
            "model-path": DummyModel()
        }

    def output(self):
        return luigi.LocalTarget("/output")


class BatchPredictTaskTest(TestCase):

    def test_task_cmd(self):
        task = DummyBatchPredictTask()
        expected = [
            "python",
            predict.__file__,
            "--runner=DataflowRunner",
            "--project=project",
            "--temp_location=temp",
            "--job_name=dummy-batch-predict",
            "--output=/output",
            "--input-data=/data/part-*",
            "--model-path=/model/",
            "--batch-size=32"
        ]
        actual = task._mk_cmd_line()
        self.assertEquals(actual, expected)
