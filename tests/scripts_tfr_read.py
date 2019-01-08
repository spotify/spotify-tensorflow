# -*- coding: utf-8 -*-
#
# Copyright 2018 Spotify AB.
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

import inspect
from os import path
import subprocess

import tensorflow as tf


# TODO(brianm): copied from examples/examples_utils.py; prefer to dedupe
def get_data_dir(subdir="train"):
    example_dir = path.dirname(inspect.stack()[0][1])
    return path.join(
        example_dir, "..", "tests",
        "resources", "tf-test-resource", "tf-records-iris", subdir)


class TFRReadTest(tf.test.TestCase):

    def test_on_path_and_executable(self):
        tf_records_path_glob = get_data_dir("eval")
        output = subprocess.check_output(["tfr-read", tf_records_path_glob + "*"])
        assert output.startswith('{"petal_length": -0.90068119764328,')
