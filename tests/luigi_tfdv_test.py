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

import luigi
from luigi.local_target import LocalTarget
from spotify_tensorflow.luigi.tfdv import TFDVGenerateStatsTask
from tensorflow.python.platform import test


class ExternalData(luigi.ExternalTask):
    def output(self):
        return LocalTarget("some/file/somewhere")


class BaseTFDVTestTask(TFDVGenerateStatsTask):
    def output(self):
        return LocalTarget("test")


class MyLocalTFDV(BaseTFDVTestTask):
    local_runner = True

    def requires(self):
        return ExternalData()


class MyTFDV(BaseTFDVTestTask):
    def requires(self):
        return ExternalData()


class LuigiTFDVTest(test.TestCase):

    @staticmethod
    def test_construct_reqs_txt():
        reqs_txt_path = TFDVGenerateStatsTask._construct_reqs_txt()
        assert open(reqs_txt_path, "r").read() == """tensorflow-transform==0.9.0
tensorflow-metadata==0.9.0
"""

    @staticmethod
    def test_stats_file_name():
        task = MyTFDV()
        assert task.stats_file_name == "_stats.pb"

    @staticmethod
    def test_python_script():
        task = MyTFDV()
        assert task.python_script.rstrip("c").endswith("tfx/tfdv.py")

    @staticmethod
    def test_requirements_file_set():
        task = MyTFDV()
        assert task.requirements_file is not None
        task = MyLocalTFDV()
        assert task.requirements_file is None

    @staticmethod
    def test_tfx_args():
        task = MyTFDV()
        assert len(task.tfx_args()) > 0
        task = MyLocalTFDV()
        assert len(task.tfx_args()) == 0

    @staticmethod
    def test_get_output_args():
        task = MyTFDV()
        output_args = task._get_output_args()
        assert len(output_args) == 1
        assert output_args[0].endswith("test/_stats.pb")
