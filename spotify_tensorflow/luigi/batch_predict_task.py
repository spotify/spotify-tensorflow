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

from abc import abstractmethod
import os
from os.path import join as pjoin

import luigi
import predict
from spotify_tensorflow.luigi.python_dataflow_task import PythonDataflowTask


class BatchPredictTask(PythonDataflowTask):
    """
    Luigi wrapper for a generic batch prediction job. Given an input folder containing tfrecord
    files and a path to a saved Tensorflow model, a Beam job will be executed to apply predictions
    to the input data.
    """
    batch_size = luigi.IntParameter(default=32, description="Number of elements per batch")

    pwd = os.path.dirname(os.path.realpath(__file__))
    python_script = predict.__file__

    @abstractmethod
    def get_input_dir(self):
        """
        This should return the GCS folder containing the input tfrecord files.
        """
        pass

    @abstractmethod
    def get_model_dir(self):
        """
        This should return the GCS folder containing a Tensorflow saved model.
        """
        pass

    def _mk_cmd_line(self):
        cmd = super(BatchPredictTask, self)._mk_cmd_line()
        cmd.extend([
            "--input-data={}".format(pjoin(self.get_input_dir(), "part-*")),
            "--model-path={}".format(self.get_model_dir()),
            "--batch-size={}".format(self.batch_size)
        ])
        return cmd

    def _get_input_args(self):
        # Note: overridden from parent class. We do not want to allow seamless input arg passing
        # here because predict.py requires specific inputs. This should not be used with a custom
        # user python script. Extra inputs from 'requires' will be ignored.
        return []
