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

from __future__ import absolute_import, division, print_function

import datetime
import getpass
import logging
import subprocess
import sys

import luigi
from luigi.contrib.gcs import GCSTarget

logger = logging.getLogger("luigi-interface")


class TensorFlowTask(luigi.Task):
    """Luigi wrapper for a TensorFlow task. To use, extend this class and provide values for the
    following properties:

    model_package = None        The name of the python package containing your model.
    model_name = None           The name of the python module containing your model.
                                Ex: if the model is in /foo/models/main.py, you would set
                                model_package = "models" and model_name = "main"
    gcp_project = None          The Google Cloud project id to run with ml-engine
    region = None               The GCP region if running with ml-engine, e.g. europe-west1
    model_name_suffix = None    A string suffix representing the model name, which will be appended
                                to the job name.

    Also, you can specify command line arguments for your trainer by overriding the
    `def tf_task_args(self)` method.
    """

    # Task properties
    model_package = None
    model_name = None
    gcp_project = None
    region = None
    model_name_suffix = None

    # Task parameters
    cloud = luigi.BoolParameter(description="Run on ml-engine")
    job_dir_prefix = luigi.Parameter(description="A file path prefix, to which a user name "
                                                 "and job name will be appended. When running on "
                                                 "ml engine, this should be a GCS uri "
                                                 "(ie. gs://bucket/path). A trailing "
                                                 "/ is not required.")
    ml_engine_conf = luigi.Parameter(default=None,
                                     description="An ml-engine YAML configuration file.")
    tf_debug = luigi.BoolParameter(default=False, description="Run tf on debug mode")

    def __init__(self, *args, **kwargs):
        super(TensorFlowTask, self).__init__(*args, **kwargs)
        self._job_name = None

    def tf_task_args(self):
        """A list of args to pass to the tf main module."""
        return []

    def run(self):
        cmd = self._mk_cmd()
        logger.info("Running:\n```\n%s\n```", cmd)
        ret = subprocess.call(cmd, shell=True)
        if ret != 0:
            sys.exit(ret)

    def _mk_cmd(self):
        cmd = ["gcloud ml-engine"]
        if self.cloud:
            cmd.extend(self._mk_cloud_params())
        else:
            cmd.append("local train")

        cmd.extend(self._get_model_args())

        if self.tf_debug:
            cmd += ["--verbosity=debug"]

        cmd.extend(self._get_job_args())
        return " ".join(cmd)

    def _mk_job_name(self):
        if not self._job_name:
            self._job_time = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
            job_name = "{user}_{time}_model".format(user=getpass.getuser(), time=self._job_time)
            model_suffix = \
                self.model_name_suffix if self.model_name_suffix else self.__class__.__name__
            job_name += "_%s" % model_suffix
            self._job_name = job_name
        return self._job_name

    def _job_dir(self):
        return "{prefix}/{user}/{job_name}".format(prefix=self.job_dir_prefix.rstrip("/"),
                                                   user=getpass.getuser(),
                                                   job_name=self._job_name)

    def _mk_cloud_params(self):
        params = []
        if self.gcp_project:
            params.append("--project=%s" % self.gcp_project)
        params.append("jobs submit training %s" % self._job_name)
        if self.region:
            params.append("--region=%s" % self.region)
        if self.ml_engine_conf:
            params.append("--config=%s" % self.ml_engine_conf)
        params.append("--job-dir=%s" % self._job_dir())
        return params

    def _get_model_args(self):
        args = []
        if self.model_package:
            package_path = "/".join(__import__(self.model_package).__file__.split("/")[:-1])
            args.append("--package-path=%s" % package_path)
        if self.model_name:
            args.append("--module-name={package}.{module}".format(package=self.model_package,
                                                                  module=self.model_name))
        return args

    def _get_job_args(self):
        args = ["--"]
        args.extend(self._get_input_args())
        if not self.cloud:
            args.append("--job-dir=%s" % self._job_dir())
        args.extend(self.tf_task_args())
        return args

    def _get_input_args(self):
        job_input = self.input()
        if isinstance(job_input, luigi.Target):
            job_input = {"input": job_input}
        if not isinstance(job_input, dict):
            raise ValueError("Input (requires()) must be dict type")
        input_args = []
        for (name, targets) in job_input.iteritems():
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
        elif isinstance(target, GCSTarget):
            return target.path
        else:
            raise ValueError("Unsupported input Target type: %s" % target.__class__.__name__)
