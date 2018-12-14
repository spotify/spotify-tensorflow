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

import logging
import tempfile

import luigi
from spotify_tensorflow.luigi.tfx_task import TFXBaseTask
from spotify_tensorflow.luigi.utils import fetch_tfdv_whl, get_uri
import spotify_tensorflow.pipeline.tfdv as tfdv_pipeline
import tensorflow_data_validation as tfdv

logger = logging.getLogger("luigi-interface")


class TFDVGenerateStatsTask(TFXBaseTask):
    """
    Thin wrapper around TensorFlow Data Validation's `tfdv.generate_statistics_from_tfrecord`.

    By default, the resulting stats is written as a `_stats.pb` file inside each respective
    input target path. This ensures the stats object is co-located with the data.
    """

    python_script = tfdv_pipeline.__file__

    def __init__(self):
        super(TFDVGenerateStatsTask, self).__init__()
        if not self.local_runner:
            self.requirements_file = TFDVGenerateStatsTask._construct_reqs_txt()

    def tfx_args(self):
        args = []
        if not self.local_runner:
            # Fetch the TFDV wheel ourselves for the Dataflow worker.
            local_tfdv_whl_path = fetch_tfdv_whl(version=tfdv.__version__)
            args.append("--extra_packages={}".format(local_tfdv_whl_path))
        return args

    @property
    def stats_file_name(self):
        return "_stats.pb"

    def _get_output_args(self):
        input_targets = luigi.task.flatten(self.input())
        assert len(input_targets) == 1, "Must provide 1 and only 1 input."
        output_path = "{}/{}".format(get_uri(input_targets[0]),
                                     self.stats_file_name)
        return ["--output={}".format(output_path)]

    @classmethod
    def _construct_reqs_txt(cls):
        """
        There are a few dependencies which are required to be installed
        on the Dataflow workers.

        This function returns a path to the requirements.txt we would like
        installed on the Dataflow workers.

        (The trick is knowing which versions to pin to.)
        """

        # Would prefer to do:
        #
        # > import tensorflow_metadata
        # > tensorflow_metadata.__version__
        #
        # But this is not available. There's an issue against upstream here:
        # https://github.com/tensorflow/metadata/issues/2
        tf_metadata_version = "0.9.0"

        # Would prefer to do:
        #
        # > import tensorflow_transform
        # > tensorflow_transform.__version__
        #
        # But this is not available. There's an issue against upstream here:
        # https://github.com/tensorflow/transform/issues/94
        #
        # For now, assume that it's either the same as TFDV or fallback to a static
        # version.
        tfdv_version = None
        try:
            import tensorflow_data_validation
            tfdv_version = tensorflow_data_validation.__version__
        except ImportError:
            pass
        tf_transform_version = tfdv_version if tfdv_version is not None else "0.11.0"

        with tempfile.NamedTemporaryFile("w", delete=False) as tfdv_reqs_txt:
            tfdv_reqs_txt.writelines([
                "tensorflow-transform=={}\n".format(tf_transform_version),
                "tensorflow-metadata=={}\n".format(tf_metadata_version)
            ])
            return tfdv_reqs_txt.name
