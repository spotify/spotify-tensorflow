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

import argparse
import logging
from os.path import join as pjoin
from typing import List  # noqa: F401

from apache_beam.options.pipeline_options import PipelineOptions
from spotify_tensorflow.luigi.utils import to_snake_case
from spotify_tensorflow.tf_schema_utils import parse_schema_txt_file
from tensorflow.python.lib.io import file_io
import tensorflow_data_validation as tfdv


class TFDV(object):

    @classmethod
    def run(cls):

        parser = argparse.ArgumentParser()
        parser.add_argument("--input")
        parser.add_argument("--output")
        args, _ = parser.parse_known_args()

        tfdv.generate_statistics_from_tfrecord(args.input, args.output,
                                               pipeline_options=PipelineOptions())


logger = logging.getLogger("luigi-interface")


class TfDataValidator:
    """Spotify-specific API for using Tensorflow Data Valiation in production.

    The typical usage is to create an instance of this class from a Luigi task that produces
    tfrecord files in order to produce statistics, a schema snapshot and any anomalies along with
    the dataset.
    """

    def __init__(self,
                 schema_path,  # type: str
                 data_location,  # type: str
                 setup_file  # type: str
                 ):
        """
        :param schema_path: tf.metadata Schema path. Must be in text format
        :param data_location: GCS folder containing tfrecord files
        :param setup_file: Path to setup.py file containing spotify-tensorflow
        """
        self.data_location = data_location
        self.schema = parse_schema_txt_file(schema_path)
        self.schema_snapshot_path = pjoin(self.data_location, "schema_snapshot.pb")
        self.stats_path = pjoin(self.data_location, "stats.pb")
        self.anomalies_path = pjoin(self.data_location, "anomalies.pb")
        # TODO: can we move this into lib?
        self.setup_file = setup_file

    def write_stats_and_schema(self, dataflow_args):  # type: (List[str]) -> None
        py_pipeline_args = ["=".join([to_snake_case(arg.split("=")[0]),
                                      arg.split("=")[1]]) for arg in dataflow_args]
        py_pipeline_args.append("--setup_file=%s" % self.setup_file)
        tfdv.generate_statistics_from_tfrecord(pjoin(self.data_location, "part-*"),
                                               output_path=self.stats_path,
                                               pipeline_options=PipelineOptions(py_pipeline_args))
        self.upload_schema()

    def validate_stats_against_schema(self):  # type: () -> bool
        stats = tfdv.load_statistics(self.stats_path)
        self.anomalies = tfdv.validate_statistics(stats, self.schema).anomaly_info.items()
        if len(self.anomalies) > 0:
            logger.error("Anomalies found in training dataset...")
            logger.error(str(self.anomalies))
            self.upload_anomalies()
            return False
        else:
            logger.info("No anomalies found")
            return True

    def upload_schema(self):  # type: () -> None
        file_io.atomic_write_string_to_file(self.schema_snapshot_path,
                                            self.schema.SerializeToString())

    def upload_anomalies(self):  # type: () -> None
        if self.anomalies:
            file_io.atomic_write_string_to_file(self.anomalies_path,
                                                self.anomalies.SerializeToString())


if __name__ == "__main__":
    TFDV.run()
