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
import sys
import time

import tensorflow_data_validation as tfdv
from apache_beam.options.pipeline_options import GoogleCloudOptions, PipelineOptions, SetupOptions
from spotify_tensorflow.tfx.utils import create_setup_file, assert_not_empty_string
from tensorflow_metadata.proto.v0 import statistics_pb2  # noqa: F401


class GenerateStats(object):
    def __init__(self, input_data, output_path=None):
        self.input_data = input_data
        if output_path is None:
            self.output_path = os.path.join(self.input_data, "_stats.pb")
        else:
            self.output_path = output_path

    def run(self, pipeline_args=None):
        if pipeline_args is None:
            pipeline_args = sys.argv[1:]
        return generate_statistics_from_tfrecord(pipeline_args=pipeline_args,
                                                 input_data_dir=self.input_data,
                                                 output_path=self.output_path)


def generate_statistics_from_tfrecord(pipeline_args,   # type: str
                                      input_data_dir,  # type: str
                                      output_path      # type: str
                                      ):
    # type: (...) ->  statistics_pb2.DatasetFeatureStatisticsList
    """
    Generate stats file from a tfrecord dataset using TFDV

    :param pipeline_args: un-parsed Dataflow arguments
    :param input_data_dir:
    :param output_path:
    :return final state of the Beam pipeline
    """
    assert_not_empty_string(input_data_dir)
    assert_not_empty_string(output_path)

    pipeline_options = PipelineOptions(flags=pipeline_args)

    all_options = pipeline_options.get_all_options()

    if all_options["job_name"] is None:
        gcloud_options = pipeline_options.view_as(GoogleCloudOptions)
        gcloud_options.job_name = "generatestats-%s" % str(time.time())[:-3]

    if all_options["setup_file"] is None:
        setup_file_path = create_setup_file()
        setup_options = pipeline_options.view_as(SetupOptions)
        setup_options.setup_file = setup_file_path

    input_files = os.path.join(input_data_dir, "*.tfrecords")
    return tfdv.generate_statistics_from_tfrecord(data_location=input_files,
                                                  output_path=output_path,
                                                  pipeline_options=pipeline_options)
