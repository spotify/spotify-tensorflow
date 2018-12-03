# -*- coding: utf-8 -*-
#
# Copyright 2017 Spotify AB.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys
from abc import abstractmethod
from typing import List  # noqa: F401

import luigi
from spotify_tensorflow.luigi.base import TensorFlowLuigiBaseTask

logger = logging.getLogger("luigi-interface")


class TFXBeamBaseTask(TensorFlowLuigiBaseTask):
    """Base task for Beam jobs"""
    runner = luigi.Parameter(default="DataflowRunner", description="Beam runner")
    project = luigi.Parameter(description="GCP project for Dataflow Beam job")
    machineWorkerType = luigi.Parameter(default="n1-standard-4", description="Dataflow worker type")
    region = luigi.Parameter(default="europe-west1", description="Dataflow region")
    temp_location = luigi.Parameter(description="Temporary location")
    python_script = luigi.Parameter(description="Python script to run Beam job")

    def __init__(self, *args, **kwargs):
        super(TFXBeamBaseTask, self).__init__(*args, **kwargs)

    def extra_cmd_line_args(self):  # type: () -> List[str]
        """Additional command line arguments specific to subtask - override to provide"""
        return []

    def run(self):
        cmd_line = self.__make_cmd_line()
        logger.info(" ".join(cmd_line))

        import subprocess
        try:
            return_code = self._run_with_logging(cmd_line)
        except subprocess.CalledProcessError as e:
            logging.error(e, exc_info=True)
            return_code = e.returncode
        sys.exit(return_code)

    def _beam_cmd_line_args(self):
        cmd_line = ["python", self.python_script]
        cmd_line.append("--runner=%s" % self.runner)
        cmd_line.append("--project=%s" % self.project)
        cmd_line.append("--temp_location=%s" % self.temp_location)
        cmd_line.append("--machineWorkerType=%s" % self.machineWorkerType)
        cmd_line.append("--region=%s" % self.region)
        return cmd_line

    def _make_cmd_line(self):
        cmd = self.__beam_cmd_line_args()
        cmd.extend(self.extra_cmd_line_args())
        return cmd


class TFTransformJob(TFXBeamBaseTask):
    """tf.transform base luigi task"""
    requirements_file = luigi.Parameter(description="Requirements file for Dataflow Beam job")

    def __init__(self, *args, **kwargs):
        super(TFTransformJob, self).__init__(*args, **kwargs)

    def get_user_args(self):  # type: () -> List[str]
        """Custom user command line arguments - override to provide"""
        return []

    @abstractmethod
    def get_schema_file(self):  # type: () -> str
        """Should return fully qualified path to the schema file."""
        pass

    def extra_cmd_line_args(self):
        cmd_line = ["--schema_file=%s" % self.get_schema_file()]
        cmd_line.append("--requirements_file=%s" % self.requirements_file)
        cmd_line.extend(self.get_user_args())
        return cmd_line
