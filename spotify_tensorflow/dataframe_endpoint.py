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
import sys
import timeit

import pandas as pd
import six
import tensorflow as tf

from .dataset import Datasets

FLAGS = tf.flags.FLAGS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DataFrameEndpoint(object):
    @classmethod
    def read_dataset(cls, dataset_path,
                     take=sys.maxsize,
                     unpack_multispec=False,
                     as_np=False,
                     feature_mapping_fn=None):
        """
        Read a TF dataset and load it into a Pandas DataFrame.

        :param dataset_path: Path to the TF Records Dataset
        :param take: Number of records to read
        :param unpack_multispec: Returns an array of DataFrames, order is the same as the one used
        when building the MultiFeatureSpec in Featran
        :param as_np: Return a numpy array (instead of a Pandas DataFrame)
        :param feature_mapping_fn: Override the TF record reading function
        :return: A Pandas DataFrame containing the dataset
        """
        return six.next(cls.batch_iterator(dataset_path, take, unpack_multispec, as_np,
                                           feature_mapping_fn))

    @classmethod
    def batch_iterator(cls, dataset_path,
                       batch_size=10000,
                       unpack_multispec=False,
                       as_np=False,
                       feature_mapping_fn=None):
        """
        Read a TF dataset in batches, each one yielded as a Pandas DataFrame.

        :param dataset_path: Path to the TF Records Dataset
        :param batch_size: Size of each batches
        :param unpack_multispec: Returns an array of DataFrames, order is the same as the one used
        when building the MultiFeatureSpec in Featran
        :param as_np: Return a numpy array (instead of a Pandas DataFrame)
        :param feature_mapping_fn: Override the TF record reading function
        :return: A Python Generator, yielding batches of data in a Pandas DataFrame
        """
        training_it, context = Datasets.mk_iter(dataset_path, feature_mapping_fn=feature_mapping_fn)
        groups = context.multispec_feature_groups if unpack_multispec else None
        for batch in cls.__DataFrameGenerator(training_it, batch_size):
            yield cls.__format_df(batch, as_np, groups)

    @staticmethod
    def __format_df(df, as_np, multispec_feature_groups):
        if not multispec_feature_groups:
            return df.as_matrix() if as_np else df
        return [df[f].as_matrix() if as_np else df[f] for f in multispec_feature_groups]

    class __DataFrameGenerator(object):
        """
        Provides a Pandas DataFrame Generator to read a TF Dataset. Created from DataFrameEndpoint.
        """

        def __init__(self, training_it, batch_size):

            self.batch_size = batch_size
            self.batch_iter = training_it.get_next()
            self.buff = None

        def __iter__(self):
            logger.info("Starting TF Session...")
            with tf.Session() as sess:
                logger.info("Reading TFRecords...")
                while True:
                    try:
                        yield self._get_batch(sess)
                    except tf.errors.OutOfRangeError:
                        logger.info("End of dataset.")
                        break
            yield self.buff

        def _get_batch(self, sess):
            if self.buff is None:
                self.buff = pd.DataFrame(sess.run(self.batch_iter))
            while len(self.buff) < self.batch_size:
                t = timeit.default_timer()
                self.buff = self.buff.append(pd.DataFrame(sess.run(self.batch_iter)))
                logger.info("Fetched %d / %s records (%4d TFExamples/s)" % (
                    len(self.buff), str(self.batch_size) if self.batch_size < sys.maxsize else "?",
                    FLAGS.batch_size / (timeit.default_timer() - t)))
            ret = self.buff[:self.batch_size]
            self.buff = self.buff[self.batch_size:]
            return ret
