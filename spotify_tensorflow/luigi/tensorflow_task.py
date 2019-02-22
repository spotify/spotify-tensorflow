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
#

from __future__ import absolute_import, division, print_function

import getpass
import logging
import uuid

import luigi
from luigi.contrib.gcs import GCSFlagTarget
from luigi.local_target import LocalTarget
from spotify_tensorflow.luigi.utils import is_gcs_path, get_uri, run_with_logging

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
    runtime_version = None      The Google Cloud ML Engine runtime version for this job. Defaults to
                                the latest stable version. See
                                https://cloud.google.com/ml/docs/concepts/runtime-version-list for a
                                list of accepted versions.
    scale_tier = None           Specifies the machine types, the number of replicas for workers and
                                parameter servers. SCALE_TIER must be one of:
                                    basic, basic-gpu, basic-tpu, custom, premium-1, standard-1.

    Also, you can specify command line arguments for your trainer by overriding the
    `def tf_task_args(self)` method.
    """

    # Task properties
    model_name = luigi.Parameter(description="Name of the python model file")
    model_package = luigi.Parameter(description="Python package containing your model")
    model_package_path = luigi.Parameter(description="Absolute path to the model package")
    gcp_project = luigi.Parameter(description="GCP project", default=None)
    region = luigi.Parameter(description="GCP region", default=None)
    model_name_suffix = luigi.Parameter(description="String which will be appended to the job"
                                                    " name. Useful for finding jobs in the"
                                                    " ml-engine UI.", default=None)

    # Task parameters
    cloud = luigi.BoolParameter(description="Run on ml-engine")
    blocking = luigi.BoolParameter(default=True, description="Run in stream-logs/blocking mode")
    job_dir = luigi.Parameter(description="A job directory, used to store snapshots, logs and any "
                                          "other artifacts. A trailing '/' is required for "
                                          "'gs://' paths.")
    ml_engine_conf = luigi.Parameter(default=None,
                                     description="An ml-engine YAML configuration file.")
    tf_debug = luigi.BoolParameter(default=False, description="Run tf on debug mode")
    runtime_version = luigi.Parameter(default=None,
                                      description="The Google Cloud ML Engine runtime version "
                                      "for this job.")
    scale_tier = luigi.Parameter(default=None,
                                 description="Specifies the machine types, the number of replicas "
                                             "for workers and parameter servers.")

    def __init__(self, *args, **kwargs):
        super(TensorFlowTask, self).__init__(*args, **kwargs)

    def tf_task_args(self):
        """A list of args to pass to the tf main module."""
        return []

    def run(self):
        cmd = self._mk_cmd()
        logger.info("Running:\n```\n%s\n```", cmd)
        run_with_logging(cmd, logger)
        logger.info("Training successful. Marking as done.")
        self._success_hook()

    def output(self):
        """
        Generate Target dynamically based on `self.get_job_dir()`.
        """
        job_dir = str(self.get_job_dir())
        return GCSFlagTarget(job_dir) if is_gcs_path(job_dir) else LocalTarget(job_dir)

    # TODO(rav): look into luigi hooks
    def _success_hook(self):
        success_file = self.get_job_dir().rstrip("/") + "/_SUCCESS"
        if is_gcs_path(self.get_job_dir()):
            from luigi.contrib.gcs import GCSClient
            client = GCSClient()
            client.put_string("", success_file)
        else:
            # assume local filesystem otherwise
            open(success_file, "a").close()

    def _mk_cmd(self):
        cmd = ["gcloud", "ml-engine"]
        if self.cloud:
            cmd.extend(self._mk_cloud_params())
        else:
            cmd.extend(["local", "train"])

        cmd.extend(self._get_model_args())

        if self.tf_debug:
            cmd.append("--verbosity=debug")

        cmd.extend(self._get_job_args())
        return cmd

    def get_job_dir(self):
        """Get job directory used to store snapshots, logs, final output and any other artifacts."""
        return self.job_dir

    def _mk_cloud_params(self):
        params = []
        if self.gcp_project:
            params.append("--project=%s" % self.gcp_project)

        params.extend(["jobs", "submit", "training", self._get_job_name()])

        if self.region:
            params.append("--region=%s" % self.region)
        if self.ml_engine_conf:
            params.append("--config=%s" % self.ml_engine_conf)
        params.append("--job-dir=%s" % self.get_job_dir())
        if self.blocking:
            params.append("--stream-logs")  # makes the execution "blocking"
        if self.runtime_version:
            params.append("--runtime-version=%s" % self.runtime_version)
        if self.scale_tier:
            params.append("--scale-tier=%s" % self.scale_tier)
        return params

    def _get_model_args(self):
        args = []
        if self.model_package_path:
            args.append("--package-path=%s" % self.model_package_path)
        if self.model_name:
            module_name = self.model_name
            if self.model_package:
                module_name = "{package}.{module}".format(package=self.model_package,
                                                          module=module_name)
            args.append("--module-name=" + module_name)
        return args

    def _get_job_args(self):
        args = ["--"]
        args.extend(self._get_input_args())
        if not self.cloud:
            args.append("--job-dir=%s" % self.get_job_dir())
        args.extend(self.tf_task_args())
        return args

    def _get_job_name(self):
        job_name = "%s_%s_%s" % (getpass.getuser(), self.__class__.__name__,
                                 str(uuid.uuid4()).replace("-", "_"))
        return job_name

    def _get_input_args(self):
        # TODO(brianm): this doesn't work when subclass yields from `requires`
        job_input = self.input()
        if isinstance(job_input, luigi.Target):
            job_input = {"input": job_input}
        if len(job_input) == 0:  # default requires()
            return []
        if not isinstance(job_input, dict):
            raise ValueError("Input (requires()) must be dict type")
        input_args = []
        for (name, targets) in job_input.items():
            uris = [get_uri(target) for target in luigi.task.flatten(targets)]
            if isinstance(targets, dict):
                # If targets is a dict that means it had multiple outputs. In this case make the
                # input args "<input key>-<task output key>"
                names = ["%s-%s" % (name, key) for key in targets.keys()]
            else:
                names = [name] * len(uris)
            for (arg_name, uri) in zip(names, uris):
                input_args.append("--%s=%s" % (arg_name, uri))

        return input_args
