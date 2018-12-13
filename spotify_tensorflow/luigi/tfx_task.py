# -*- coding: utf-8 -*-
#
# Copyright 2017 Spotify AB.
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

from spotify_tensorflow.luigi.python_dataflow_task import PythonDataflowTask


class TFXBaseTask(PythonDataflowTask):
    def __init__(self, *args, **kwargs):
        super(TFXBaseTask, self).__init__(*args, **kwargs)
        if self.job_name is None:
            # job_name must consist of only the characters [-a-z0-9]
            self.job_name = self.__class__.__name__.replace("_", "-").lower()

    def tfx_args(self):
        """ Extra arguments that will be passed to your tfx dataflow job.

        Example:
            return ["--schema_file=gs://uri/to/schema_file"]
        Note that:

            * You "set" args by overriding this method in your tfx subclass.
            * This function should return an iterable of strings.
        """
        return []

    def _mk_cmd_line(self):
        cmd_line = super(TFXBaseTask, self)._mk_cmd_line()
        cmd_line.extend(self.tfx_args())
        return cmd_line
