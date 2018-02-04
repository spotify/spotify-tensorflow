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

import tensorflow as tf

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FreezeGraph(object):

    @classmethod
    def session(cls, session, path, network):
        """
        Freeze a graph by taking a session and a network and storing
        the results into a pb file at the given path. This function wil convert
        variables to constants which is necessary for JVM serving.

        :param session: TF Session
        :param path: Where the graph will be written
        :param network: Tensor representing a network(Usually a prediction)
        :return: Path to the written graph
        """
        input_graph_def = tf.get_default_graph().as_graph_def()

        logger.info("Freezing model")
        output_node_names = [t.op.name for t in [network]]

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            session,
            input_graph_def,
            output_node_names,
            variable_names_blacklist=["global_step"]
        )

        with tf.gfile.GFile(path, "wb") as f:
            f.write(output_graph_def.SerializeToString())

        return path

    @classmethod
    def checkpoint(cls, checkpoint, path, network):
        """
        Freeze a graph by taking a checkpoint and a network and storing
        the results into a pb file at the given path. This function wil convert
        variables to constants which is necessary for JVM serving.

        :param checkpoint: Path to a local checkpoint file
        :param path: Where the graph will be written
        :param network: Tensor representing a network(Usually a prediction)
        :return: Path to the written graph
        """
        with tf.Session(graph=tf.Graph()) as sess:
            saver = tf.train.import_meta_graph(checkpoint + ".meta", clear_devices=True)
            saver.restore(sess, checkpoint)
            return cls.session(sess, path, network)
