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

import sys
import time

import luigi
from luigi.cmdline import luigi_run
from luigi.contrib.gcs import GCSFlagTarget, GCSTarget
from spotify_tensorflow.luigi.tfdv import TFDVGenerateStatsTask
from spotify_tensorflow.luigi.utils import get_uri


class TaxiDataSmall(luigi.ExternalTask):
    def output(self):
        return GCSTarget("gs://sp-ml-infra-public/datasets/chicago-taxi-trips/small")


class TaxiDataLarge(luigi.ExternalTask):
    def output(self):
        return GCSTarget("gs://sp-ml-infra-public/datasets/chicago-taxi-trips/full")


class TaxiStats(TFDVGenerateStatsTask):

    runner = "DataflowRunner"
    project = "sp-ml-infra"
    temp_location = "gs://sp-ml-infra-public/tmp"
    local_runner = True

    def requires(self):
        return TaxiDataSmall()

    def output(self):
        return GCSFlagTarget("{}/{}-{}/".format(
            get_uri(self.input()),
            "-stats-",
            str(time.time())))


def main():
    luigi_run(sys.argv[1:] + [
        "--local-scheduler",
        "TaxiStats"])


if __name__ == "__main__":
    main()
