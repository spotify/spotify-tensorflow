# -*- coding: utf-8 -*-
#
#  Copyright 2018 Spotify AB.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.

import os
import subprocess
from os.path import dirname, abspath

import tensorflow as tf

SUCCESS_EXIT = 0


class FlagsTest(tf.test.TestCase):

    _this_dir = os.path.dirname(os.path.abspath(__file__))

    def test_legacy_flags(self):
        assert self._run_dummy_model("--job_dir=gs://foo/bar") != SUCCESS_EXIT
        assert self._run_dummy_model("--job_dir gs://foo/bar") != SUCCESS_EXIT

    def test_expected_flags(self):
        assert self._run_dummy_model("--job-dir gs://foo/bar") == SUCCESS_EXIT
        assert self._run_dummy_model("--job-dir=gs://foo/bar") == SUCCESS_EXIT

    def test_unknown_flags(self):
        assert self._run_dummy_model("--unknwn-flag=foo") == SUCCESS_EXIT

    @staticmethod
    def _run_dummy_model(args):
        FNULL = open(os.devnull, "w")
        d = dirname(dirname(abspath(__file__)))
        return subprocess.call("PYTHONPATH=%s:$PYTHONPATH python %s %s" % (d, __file__, args),
                               stderr=FNULL,
                               shell=True)


def dummy_model(_):
    # trigger flags resolution
    from spotify_tensorflow import flags  # noqa: F401


if __name__ == "__main__":
    tf.app.run(main=dummy_model)
