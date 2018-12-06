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
from luigi.contrib.gcs import GCSTarget, GCSFlagTarget

logger = logging.getLogger("luigi-interface")


class BeamBaseTask(luigi.Task):
    """Base task for Beam jobs"""
    runner = luigi.Parameter(default="DataflowRunner", description="Beam runner")
    project = luigi.Parameter(description="GCP project for Dataflow Beam job")
    machineWorkerType = luigi.Parameter(default="n1-standard-4", description="Dataflow worker type")
    region = luigi.Parameter(default="europe-west1", description="Dataflow region")
    temp_location = luigi.Parameter(description="Temporary location")

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

    def _get_input_args(self):
        job_input = self.input()
        if isinstance(job_input, luigi.Target):
            job_input = {"input": job_input}
        if len(job_input) == 0:  # default requires()
            return []
        if not isinstance(job_input, dict):
            raise ValueError("Input (requires()) must be dict type")
        input_args = []
        for (name, targets) in job_input.items():
            uris = [self._get_uri(target) for target in luigi.task.flatten(targets)]
            if isinstance(targets, dict):
                # If targets is a dict that means it had multiple outputs. In this case make the
                # input args "<input key>-<task output key>"
                names = ["%s-%s" % (name, key) for key in targets.keys()]
            else:
                names = [name] * len(uris)
            for (arg_name, uri) in zip(names, uris):
                input_args.append("--%s=%s" % (arg_name, uri))

        return input_args

    @staticmethod
    def _get_uri(target):
        if hasattr(target, "uri"):
            return target.uri()
        elif isinstance(target, (GCSTarget, GCSFlagTarget)):
            return target.path
        else:
            raise ValueError("Unsupported input Target type: %s" % target.__class__.__name__)

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
