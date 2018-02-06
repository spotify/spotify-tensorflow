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

from __future__ import absolute_import, division, print_function

import os
import tempfile

import tensorflow as tf
from spotify_tensorflow.freeze_graph import FreezeGraph
from tensorflow.python.platform import gfile


class FreezeTest(tf.test.TestCase):
    def setUp(self):
        self.output_file = tempfile.NamedTemporaryFile(delete=False).name

    def tearDown(self):
        os.unlink(self.output_file)

    def test_freeze_session(self):
        with self.test_session() as sess:
            x = tf.square([2, 3])
            FreezeGraph.session(sess, self.output_file, x)
            self.assertTrue(os.path.isfile(self.output_file))

    def test_read_freeze_session(self):
        with self.test_session():
            with gfile.FastGFile(self.output_file, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def)
