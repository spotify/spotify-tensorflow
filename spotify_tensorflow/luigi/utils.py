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

import re
import os
import subprocess
import tempfile

import requests


def to_snake_case(s, sep="_"):
    # type: (str, str) -> str
    p = r"\1" + sep + r"\2"
    s1 = re.sub("(.)([A-Z][a-z]+)", p, s)
    return re.sub("([a-z0-9])([A-Z])", p, s1).lower()


def is_gcs_path(path):
    # type: (str) -> bool
    """Returns True if given path is GCS path, False otherwise."""
    return path.strip().lower().startswith("gs://")


def get_uri(target):
    if hasattr(target, "uri"):
        return target.uri()
    elif hasattr(target, "path"):
        return target.path
    else:
        raise ValueError("Unknown input target type: %s" % target.__class__.__name__)


def run_with_logging(cmd, logger):
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


def _fetch_file(url, output_path=None):
    # type: (str, str) -> str
    """Fetches a file from the url and saves it to a temp file (or at the provided output path)."""
    rep = requests.get(url, allow_redirects=True)
    if rep.status_code / 100 != 2:
        raise Exception("Got [status_code:{}] fetching file at [url:{}]".format(rep.status_code,
                                                                                url))

    if output_path is None:
        output_path = tempfile.NamedTemporaryFile(delete=False).name

    with open(output_path, "wb") as out:
        out.write(rep.content)

    return output_path


def fetch_tfdv_whl(version=None, output_path=None, platform="manylinux1"):
    # type: (str, str, str) -> str
    """Fetches the TFDV pip package from PyPI and saves it to a temporary file (or the provided
    output path). Returns the path to the fetched package."""
    package_name = "tensorflow_data_validation"

    if version is None:
        import tensorflow_data_validation as tfdv
        version = tfdv.__version__

    pypi_base = "https://pypi.org/simple/{}".format(package_name)

    package_url = None
    with open(_fetch_file(pypi_base)) as listing_html:
        for line in listing_html:
            if version in line and platform in line:
                package_url = re.findall(".*href=\"([^ ]*)#[^ ]*\".*", line)[0]
                break

    if package_url is None:
        raise Exception("Problem fetching package. Couldn't parse listing at [url:{}]"
                        .format(pypi_base))

    if output_path is None:
        temp_dir = tempfile.mkdtemp()
        # Note: output_path file name must exactly match the remote wheel name.
        output_path = os.path.join(temp_dir, package_url.split("/")[-1])

    return _fetch_file(package_url, output_path=output_path)
