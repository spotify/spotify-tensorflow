import logging

import tensorflow as tf

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FreezeGraph(object):

    @classmethod
    def freeze_graph_session(session, path, network):
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
            variable_names_blacklist=['global_step']
        )

        with tf.gfile.GFile(path, "wb") as f:
            f.write(output_graph_def.SerializeToString())

        return path

    @classmethod
    def freeze_graph_checkpoint(cls, checkpoint, path, network):
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
            saver = tf.train.import_meta_graph(checkpoint + '.meta', clear_devices=True)
            saver.restore(sess, checkpoint)
            return cls.freeze_graph_session(sess, path, network)
