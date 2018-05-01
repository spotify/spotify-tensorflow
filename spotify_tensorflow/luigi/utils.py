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


def to_snake_case(s, sep="_"):
    # type: (str, str) -> str
    p = r"\1" + sep + r"\2"
    s1 = re.sub("(.)([A-Z][a-z]+)", p, s)
    return re.sub("([a-z0-9])([A-Z])", p, s1).lower()


def is_gcs_path(path):
    # type: (str) -> bool
    """Returns True if given path is GCS path, False otherwise."""
    return path.strip().lower().startswith("gs://")
