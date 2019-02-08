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
import tempfile
import textwrap
from typing import Any  # noqa: F401


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
    contents_for_setup_file = """
    import setuptools
    
    if __name__ == "__main__":
        setuptools.setup(
            name="spotify_tensorflow_dataflow",
            packages=setuptools.find_packages(),
            install_requires=[
                "spotify-tensorflow[tfdv]"
        ])
    """  # noqa: W293
    setup_file_path = os.path.join(tempfile.mkdtemp(), "setup.py")
    with open(setup_file_path, "w") as f:
        f.writelines(textwrap.dedent(contents_for_setup_file))
    return setup_file_path
