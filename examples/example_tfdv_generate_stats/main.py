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

from examples.examples_utils import get_taxi_data_dir
from spotify_tensorflow.tfx.tfdv import GenerateStats

if __name__ == "__main__":

    taxi_data = get_taxi_data_dir()
    tmp_dir = tempfile.mkdtemp()
    pipeline_args = [
        "--temp_location=%s" % tmp_dir,
        "--staging_location=%s" % tmp_dir,
        "--runner=DirectRunner"
    ]

    GenerateStats(taxi_data).run(pipeline_args)
    os.remove(os.path.join(taxi_data, "stats.pb"))
