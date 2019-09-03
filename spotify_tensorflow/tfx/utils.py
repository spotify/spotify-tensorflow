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
import os
from pbr.version import VersionInfo
import tempfile
import textwrap
from typing import Any  # noqa: F401

from spotify_tensorflow.luigi.utils import to_snake_case


def assert_not_none(arg):
    # type: (Any) -> None
    if arg is None:
        raise TypeError("Argument can't be a None")


def assert_not_empty_string(arg):
    # type: (Any) -> None
    if not isinstance(arg, str):
        raise TypeError("Argument should be a string")
    if arg == "":
        raise ValueError("Argument can't be an empty string")


def create_setup_file():
    lib_version = VersionInfo("spotify_tensorflow").version_string()
    contents_for_setup_file = """
    import setuptools
    
    if __name__ == "__main__":
        setuptools.setup(
            name="spotify_tensorflow_dataflow",
            packages=setuptools.find_packages(),
            install_requires=[
                "spotify-tensorflow=={version}"
        ])
    """.format(version=lib_version)  # noqa: W293
    setup_file_path = os.path.join(tempfile.mkdtemp(), "setup.py")
    with open(setup_file_path, "w") as f:
        f.writelines(textwrap.dedent(contents_for_setup_file))
    return setup_file_path


def clean_up_pipeline_args(pipeline_args):
    output_args = list()
    for arg in pipeline_args:
        if arg.startswith("--"):
            if "=" in arg:
                k, v = arg.split("=")
                output_args.extend([to_snake_case(k), v])
            else:
                output_args.append(to_snake_case(arg))
        else:
            output_args.append(arg)

    keys = output_args[0::2]
    vals = output_args[1::2]
    return ["%s=%s" % (key, val) for (key, val) in zip(keys, vals)
            if key in SUPPORTED_DATAFLOW_PIPELINE_ARGS]


SUPPORTED_DATAFLOW_PIPELINE_ARGS = {
    "--runner",
    "--project",
    "--staging_location",
    "--zone",
    "--region",
    "--temp_location",
    "--num_workers",
    "--autoscaling_algorithm",
    "--max_num_workers",
    "--network",
    "--subnetwork",
    "--disk_size_gb",
    "--worker_machine_type",
    "--job_name",
    "--worker_disk_type",
    "--service_account_email",
    "--requirements_file"
    "--setup_file"
}
