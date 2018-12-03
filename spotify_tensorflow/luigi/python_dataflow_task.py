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

import logging
import os
import subprocess

import luigi
from luigi.task import MixinNaiveBulkComplete
from spotify_tensorflow.luigi.utils import get_uri

logger = logging.getLogger("luigi-interface")


class PythonDataflowTask(MixinNaiveBulkComplete, luigi.Task):
    """"Luigi wrapper for a dataflow job

    The following properties can be set:
    python_script = None            # Python script for the dataflow task.
    project = None                  # Name of the project owning the dataflow job.
    staging_location = None         # GCS path for staging code packages needed by workers.
    zone = None                     # GCE availability zone for launching workers.
    region = None                   # GCE region for creating the dataflow job.
    temp_location = None            # GCS path for saving temporary workflow jobs.
    num_workers = None              # The number of workers to start the task with.
    autoscaling_algorithm = None    # Set to "NONE" to disable autoscaling. `num_workers`
                                    # will then be used for the job.
    max_num_workers = None          # Used if the autoscaling is enabled.
    network = None                  # Network in GCE to be used for launching workers.
    subnetwork = None               # Subnetwork in GCE to be used for launching workers.
    disk_size_gb = None             # Remote worker disk size, if not defined uses default size.
    worker_machine_type = None      # Machine type to create Dataflow worker VMs. If unset,
                                    # the Dataflow service will choose a reasonable default.
    worker_disk_type = None         # Specify SSD for local disk or defaults to hard disk.
    service_account = None          # Service account of Dataflow VMs/workers. Default is a
                                      default GCE service account.
    job_name = None                 # Name of the dataflow job
    requirements_file = None        # Path to a requirements file containing package dependencies.
    local_runner = False            # If local_runner = True, the job uses DirectRunner,
                                      otherwise it uses DataflowRunner

    :Example:

    class AwesomeJob(PythonDataflowJobTask):
        python_script = "/path/to/python_script"
        project = "gcp-project"
        staging_location = "gs://gcp-project-playground/user/staging"
        temp_location = "gs://gcp-project-playground/user/tmp"
        max_num_workers = 20
        region = "europe-west1"
        service_account_email = "service_account@gcp-project.iam.gserviceaccount.com"

        def output(self):
            ...
    """
    # Required dataflow args
    python_script = None  # type: str
    project = None  # type: str
    staging_location = None  # type: str

    # Dataflow requires one and only one of:
    zone = None  # type: str
    region = None  # type: str

    # Optional dataflow args
    temp_location = None  # type: str
    num_workers = None  # type: str
    autoscaling_algorithm = None  # type: str
    max_num_workers = None  # type: str
    network = None  # type: str
    subnetwork = None  # type: str
    disk_size_gb = None  # type: str
    worker_machine_type = None  # type: str
    worker_disk_type = None  # type: str
    service_account = None  # type: str
    job_name = None  # type: str
    requirements_file = None  # type: str
    local_runner = False  # type: bool

    def __init__(self, *args, **kwargs):
        super(PythonDataflowTask, self).__init__(*args, **kwargs)
        self._output = self.output()
        if isinstance(self._output, luigi.Target):
            self._output = {"output": self._output}

    def on_successful_run(self):
        """ Callback that gets called right after the dataflow job has finished successfully but
        before validate_output is run.
        """
        pass

    def validate_output(self):
        """ Callback that can be used to validate your output before it is moved to it's final
        location. Returning false here will cause the job to fail, and output to be removed instead
        of published.

        :return: Whether the output is valid or not
        :rtype: Boolean
        """
        return True

    def file_pattern(self):
        """ If one/some of the input target files are not in the pattern of part-*,
        we can add the key of the required target and the correct file pattern
        that should be appended in the command line here. If the input target key is not found
        in this dict, the file pattern will be assumed to be part-* for that target.

        :return A dictionary of overrided file pattern that is not part-* for the inputs
        :rtype: Dict of String to String
        """
        return {}

    def run(self):
        cmd_line = self._mk_cmd_line()
        logger.info(" ".join(cmd_line))

        try:
            run_with_logging(cmd_line)
        except subprocess.CalledProcessError as e:
            logging.error(e, exc_info=True)
            # exit luigi with the same exit code as the python dataflow job proccess
            # In this way users can easily exit the job with code 50 to avoid Styx retries
            # https://github.com/spotify/styx/blob/master/doc/design-overview.md#workflow-state-graph
            os._exit(e.returncode)

        self.on_successful_run()
        if self.validate_output():
            self._publish_outputs()
        else:
            raise ValueError("Output is not valid")

    def _publish_outputs(self):
        for (name, target) in self._output.items():
            if hasattr(target, "publish"):
                target.publish(self._output_uris[name])

    def _mk_cmd_line(self):
        cmd_line = self._dataflow_executable()
        cmd_line.extend(self._get_dataflow_args())
        cmd_line.extend(self._get_input_args())
        cmd_line.extend(self._get_output_args())
        cmd_line.extend(self.args())
        return cmd_line

    def _dataflow_executable(self):
        """
        Defines the executable used to run the python dataflow job.
        """
        return ["python", self.python_script]

    def _get_input_uri(self, file_pattern, target):
        uri = get_uri(target)
        uri = uri.rstrip("/") + "/" + file_pattern
        return uri

    def _get_file_pattern(self):
        file_pattern = self.file_pattern()
        if not isinstance(file_pattern, dict):
            raise ValueError("file_pattern() must return a dict type")
        return file_pattern

    def _get_input_args(self):
        """
        Collects outputs from requires() and converts them to input arguments.
        file_pattern() is called to construct input file path glob with default value "part-*"
        """
        job_input = self.input()
        if isinstance(job_input, luigi.Target):
            job_input = {"input": job_input}
        if not isinstance(job_input, dict):
            raise ValueError("Input (requires()) must be dict type")

        input_args = []
        file_pattern_dict = self._get_file_pattern()
        for (name, targets) in job_input.items():
            uri_targets = luigi.task.flatten(targets)
            pattern = file_pattern_dict.get(name, "part-*")
            uris = [self._get_input_uri(pattern, uri_target) for uri_target in uri_targets]
            if isinstance(targets, dict):
                # If targets is a dict that means it had multiple outputs.
                #  Make the input args in that case "<input key>-<task output key>"
                names = ["%s-%s" % (name, key) for key in targets.keys()]
            else:
                names = [name] * len(uris)
            for (arg_name, uri) in zip(names, uris):
                input_args.append("--%s=%s" % (arg_name, uri))

        return input_args

    def _get_output_args(self):
        if not isinstance(self._output, dict):
            raise ValueError("Output must be dict type")

        output_args = []
        self._output_uris = {}

        for (name, target) in self._output.items():
            uri = target.generate_uri() if hasattr(target, "generate_uri") else get_uri(target)
            uri = uri.rstrip("/")
            output_args.append("--%s=%s" % (name, uri))
            self._output_uris[name] = uri

        return output_args

    def _get_runner(self):
        return "DirectRunner" if self.local_runner else "DataflowRunner"

    def _get_dataflow_args(self):
        dataflow_args = []

        _runner = self._get_runner()
        if _runner:
            dataflow_args += ["--runner={}".format(_runner)]
        if self.project:
            dataflow_args += ["--project={}".format(self.project)]
        if self.staging_location:
            dataflow_args += ["--staging_location={}".format(self.staging_location)]
        if self.zone:
            dataflow_args += ["--zone={}".format(self.zone)]
        if self.region:
            dataflow_args += ["--region={}".format(self.region)]
        if self.temp_location:
            dataflow_args += ["--temp_location={}".format(self.temp_location)]
        if self.num_workers:
            dataflow_args += ["--num_workers={}".format(self.num_workers)]
        if self.autoscaling_algorithm:
            dataflow_args += ["--autoscaling_algorithm={}".format(self.autoscaling_algorithm)]
        if self.max_num_workers:
            dataflow_args += ["--max_num_workers={}".format(self.max_num_workers)]
        if self.network:
            dataflow_args += ["--network={}".format(self.network)]
        if self.subnetwork:
            dataflow_args += ["--subnetwork={}".format(self.subnetwork)]
        if self.disk_size_gb:
            dataflow_args += ["--disk_size_gb={}".format(self.disk_size_gb)]
        if self.worker_machine_type:
            dataflow_args += ["--worker_machine_type={}".format(self.worker_machine_type)]
        if self.job_name:
            dataflow_args += ["--job_name={}".format(self.job_name)]
        if self.worker_disk_type:
            dataflow_args += ["--worker_disk_type={}".format(self.worker_disk_type)]
        if self.service_account:
            dataflow_args += ["--service_account_email={}".format(self.service_account)]
        if self.requirements_file:
            dataflow_args += ["--requirements_file={}".format(self.requirements_file)]

        return dataflow_args

    def args(self):
        """ Extra arguments that will be passed to your dataflow job.

        Example:
            return ["--project=my-gcp-project",
                    "--zone=a-zone",
                    "--staging_location=gs://my-gcp-project/dataflow"]

        Note that:

            * You "set" args by overriding this method in your subclass.
            * This function should return an iterable of strings.
        """
        return []

    def get_output_uris(self):
        """ Returns a dictionary that contains output uris.
        The key is the name of the output target defined in output(), and the value is
        the path/uri of the output target. It can be used to write data to different sub directories
        under one output target.

        :return A dictionary of output uris
        :rtype: Dict of String to String
        """
        return self._output_uris


def run_with_logging(cmd):
    """
    Run cmd and wait for it to finish. While cmd is running, we read it's
    output and print it to a logger.
    """
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
