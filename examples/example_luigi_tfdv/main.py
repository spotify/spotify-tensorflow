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

import sys
import time

import luigi
import tensorflow_data_validation as tfdv

from apache_beam.options.pipeline_options import GoogleCloudOptions, PipelineOptions, \
    SetupOptions, StandardOptions
from luigi.cmdline import luigi_run
from luigi.contrib.gcs import GCSTarget

from spotify_tensorflow.luigi.tfdv import TFRecordsStatsTask
from spotify_tensorflow.luigi.utils import fetch_tfdv_whl


class TaxiDataSmall(luigi.ExternalTask):
    def output(self):
        return GCSTarget("gs://ml-sketchbook-paved-road/datasets/chicago-taxi-trips/small"
                         "/20181109T215136.273716-da63ad8a7531")


class TaxiDataLarge(luigi.ExternalTask):
    def output(self):
        return GCSTarget("gs://ml-sketchbook-paved-road/datasets/chicago-taxi-trips/full/2018-11-08"
                         "/20181109T170148.261696-8f7dfbdda381")


class TaxiStats(TFRecordsStatsTask):

    def input(self):
        # TODO(brianm): ensure we support yield here
        return TaxiDataSmall().output()

    def run(self):
        pipeline_options = PipelineOptions()

        google_cloud_options = pipeline_options.view_as(GoogleCloudOptions)
        google_cloud_options.project = 'ml-sketchbook'
        google_cloud_options.job_name = 'taxi-states-brianm-' + str(time.time())[:-3]
        google_cloud_options.staging_location = 'gs://ml-sketchbook-brianm/staging'
        google_cloud_options.temp_location = 'gs://ml-sketchbook-brianm/temp'

        standard_options = pipeline_options.view_as(StandardOptions)
        standard_options.runner = 'DataflowRunner'

        import tempfile
        with tempfile.NamedTemporaryFile('w', delete=False) as tfdv_reqs_txt:
            tfdv_reqs_txt_path = tfdv_reqs_txt.name
            tfdv_reqs_txt.writelines([
                'tensorflow-transform\n',
                'tensorflow-metadata\n'
            ])

        local_tfdv_whl_path = fetch_tfdv_whl(version=tfdv.__version__)

        setup_options = pipeline_options.view_as(SetupOptions)
        setup_options.requirements_file = tfdv_reqs_txt_path
        setup_options.extra_packages = [local_tfdv_whl_path]

        self.pipeline_options = pipeline_options
        super(TaxiStats, self).run()


def main():
    argv = list(sys.argv[1:])
    argv.extend([
        '--local-scheduler',
        'TaxiStats'
    ])

    luigi_run(argv)


if __name__ == '__main__':
    main()
