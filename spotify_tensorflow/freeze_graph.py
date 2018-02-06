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
import timeit

import tensorflow as tf
from tensorflow.python.lib.io import file_io

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
        :param network: Tensor, Operation name, or list of Operation names
        :return: Path to the written graph
        """
        input_graph_def = tf.get_default_graph().as_graph_def()

        time = timeit.default_timer()
        logger.info("Freezing model at {}".format(time))

        output_node_names = None
        if type(network) == tf.Tensor:
            output_node_names = [t.op.name for t in [network]]
        elif type(network) == str:
            output_node_names = [network]
        elif type(network) == list:
            output_node_names = network
        else:
            raise ValueError("Network must be a Tensor, String or List of Strings")

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            session,
            input_graph_def,
            output_node_names,
            variable_names_blacklist=["global_step"]
        )

        file_io.write_string_to_file(path, output_graph_def.SerializeToString())

        logger.info("Froze graph in %4d seconds" % (timeit.default_timer() - time))

        return path

    @classmethod
    def checkpoint(cls, checkpoint, path, network):
        """
        Freeze a graph by taking a checkpoint and a network and storing
        the results into a pb file at the given path. This function will convert
        variables to constants which is necessary for JVM serving.

        :param checkpoint: Path to a local checkpoint file
        :param path: Where the graph will be written
        :param network: Tensor, Operation name, or list of Operation names
        :return: Path to the written graph
        """
        with tf.Session(graph=tf.Graph()) as sess:
            saver = tf.train.import_meta_graph(checkpoint + ".meta", clear_devices=True)
            saver.restore(sess, checkpoint)
            return cls.session(sess, path, network)
