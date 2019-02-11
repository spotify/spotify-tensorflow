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
from os.path import join as pjoin

from examples.examples_utils import get_taxi_data_dir
from spotify_tensorflow.tfx.tfdv import TfDataValidator

if __name__ == "__main__":

    taxi_data = get_taxi_data_dir()
    tmp_dir = tempfile.mkdtemp()
    schema = pjoin(taxi_data, "chicago_taxi_schema.pbtxt")
    schema_snapshot_path = pjoin(taxi_data, "schema_snapshot.pb")
    stats_file = pjoin(taxi_data, "stats.pb")
    anomalies_path = pjoin(taxi_data, "anomalies.pb")

    pipeline_args = [
        "--temp_location=%s" % tmp_dir,
        "--staging_location=%s" % tmp_dir,
        "--runner=DirectRunner"
    ]

    validator = TfDataValidator(schema, taxi_data)
    validator.write_stats_and_schema(pipeline_args)
    validator.validate_stats_against_schema()
    os.remove(stats_file)
    os.remove(schema_snapshot_path)
    os.remove(anomalies_path)
