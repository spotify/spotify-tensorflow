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

import abc
import logging
import time

import luigi
import six
import tensorflow_data_validation as tfdv
from apache_beam.io.filesystems import FileSystems
from apache_beam.options.pipeline_options import PipelineOptions
from luigi.contrib.gcs import GCSTarget
from tensorflow_data_validation.statistics import stats_options

# TODO(brianm): Is this right? Shouldn't we be failing if beam isn't installed?
try:
    from apache_beam.io.localfilesystem import LocalFileSystem
except ImportError:
    pass

try:
    from apache_beam.io.gcp.gcsfilesystem import GCSFileSystem
except ImportError:
    pass

logger = logging.getLogger()


# TODO(brianm): This should use the BeamBaseTask when ready.
@six.add_metaclass(abc.ABCMeta)
class TFRecordsStatsTask(luigi.Task):
    """Base luigi task used to compute TensorFlow statistics.

    Thin wrapper around TensorFlow Data Validation's `tfdv.generate_statistics_from_tfrecord`.

    # TODO(brianm):
    # By default, the resulting stats is written as a `_stats.pb` file inside each respective path
    # returned by the `tfrecords_path` method. This way, the stats object is co-located with the
    # data.
    """
    stats_options = stats_options.StatsOptions()
    pipeline_options = None  # type: pipeline_options.PipelineOptions

    @property
    def stats_file_name(self):
        return "_stats.pb"

    def run(self):
        # TODO(brianm): prefer not to do this manipulation.
        input_target_path = self._target_to_path(self.input())
        if not input_target_path.endswith("*.tfrecords"):
            if not input_target_path.endswith("/"):
                input_target_path += "/"
            input_target_path += "*.tfrecords"

        # TODO(brianm): just initialize above, ala stats_options?
        pipeline_options = self.pipeline_options
        if pipeline_options is None:
            pipeline_options = PipelineOptions()

        tfdv.generate_statistics_from_tfrecord(input_target_path, self.output().path,
                                               stats_options=self.stats_options,
                                               pipeline_options=pipeline_options)

    # def complete(self):
    #     # TODO(brianm): I don't understand Romain's comment here.
    #     # Unless overridden, the task's output depends on input's output
    #     inputs = luigi.task.flatten(self.input())
    #     if not all(i.exists() for i in inputs):
    #         return False
    #     return super(TFRecordsStatsTask, self).complete()

    def _path_to_target(self, path):
        fs = FileSystems.get_filesystem(path)
        if isinstance(fs, GCSFileSystem):
            return luigi.contrib.gcs.GCSTarget(path)
        if isinstance(fs, LocalFileSystem):
            return luigi.LocalTarget(path)
        raise ValueError("Unsupported scheme {}. Please submit and issue/patch or override "
                         "the output method.".format(FileSystems.get_scheme(path)))

    def _target_to_path(self, target):
        if hasattr(target, 'uri'):
            return target.uri()
        if hasattr(target, 'path'):
            # TODO(brianm): GCSTarget has this as a property, do some targets have it as a callable?
            return target.path
        raise ValueError('Unsupported Target type %s' % target.__class__.__name__)

    def output(self):
        input_targets = luigi.task.flatten(self.input())
        assert len(input_targets) == 1, "Must provide 1 and only 1 input."

        output_path = "{}/{}-{}".format(self._target_to_path(input_targets[0]),
                                        self.stats_file_name,
                                        str(time.time()))

        output_target = self._path_to_target(output_path)
        return output_target
