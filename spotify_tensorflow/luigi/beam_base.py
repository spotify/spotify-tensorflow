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
from typing import List  # noqa: F401

import luigi

logger = logging.getLogger("luigi-interface")


class BeamBaseTask(luigi.Task):
    """Base task for Beam jobs"""
    runner = luigi.Parameter(default="DataflowRunner", description="Beam runner")
    project = luigi.Parameter(description="GCP project for Dataflow Beam job")
    machineWorkerType = luigi.Parameter(default="n1-standard-4", description="Dataflow worker type")
    region = luigi.Parameter(default="europe-west1", description="Dataflow region")
    temp_location = luigi.Parameter(description="Temporary location")

    def __init__(self, *args, **kwargs):
        super(BeamBaseTask, self).__init__(*args, **kwargs)

    def extra_cmd_line_args(self):  # type: () -> List[str]
        """
        Additional command line arguments specific to subtask.
        Override to provide.
        """
        return []

    def run(self):
        cmd_line = self._make_cmd_line()
        logger.info(" ".join(cmd_line))

        import subprocess
        try:
            return_code = self._run_with_logging(cmd_line)
        except subprocess.CalledProcessError as e:
            logging.error(e, exc_info=True)
            return_code = e.returncode
        sys.exit(return_code)

    def beam_executable(self):
        """
        Defines the executable used to run the beam job.
        Override to provide.
        :return:
        """
        pass

    def _beam_cmd_line_args(self):
        cmd_line = [self.beam_executable()]

        cmd_line.append("--runner=%s" % self.runner)
        cmd_line.append("--project=%s" % self.project)
        cmd_line.append("--temp_location=%s" % self.temp_location)
        cmd_line.append("--machineWorkerType=%s" % self.machineWorkerType)
        cmd_line.append("--region=%s" % self.region)

        return cmd_line

    def _make_cmd_line(self):
        cmd = self._beam_cmd_line_args()
        cmd.extend(self.extra_cmd_line_args())
        return cmd

    @staticmethod
    def _run_with_logging(cmd):
        """
        Run cmd and wait for it to finish. While cmd is running, we read it's
        output and print it to a logger.
        """
        import subprocess
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output_lines = []
        while True:
            line = process.stdout.readline()
            if not line:
                break
            line = line.decode("utf-8")
            output_lines += [line]
            logger.info(line.rstrip("\n"))
        exit_code = process.wait()
        if exit_code:
            output = "".join(output_lines)
            raise subprocess.CalledProcessError(exit_code, cmd, output=output)
        return exit_code


class PythonBeamBaseTask(BeamBaseTask):
    python_script = luigi.Parameter(description="Python script to run Beam job")

    def __init__(self, *args, **kwargs):
        super(PythonBeamBaseTask, self).__init__(*args, **kwargs)

    def beam_executable(self):
        cmd_line = "python {s}".format(s=self.python_script)
        return cmd_line
