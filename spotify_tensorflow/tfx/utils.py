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
import tempfile
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


def construct_tft_reqs_txt():
    # type: () -> str
    tf_metadata_version = "0.9.0"
    tf_transform_version = "0.9.0"

    with tempfile.NamedTemporaryFile("w", delete=False) as tft_reqs_txt:
        tft_reqs_txt.writelines([
            "tensorflow-transform=={}\n".format(tf_transform_version),
            "tensorflow-metadata=={}\n".format(tf_metadata_version)
        ])
        return tft_reqs_txt.name
