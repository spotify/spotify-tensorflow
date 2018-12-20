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

from __future__ import absolute_import, division, print_function

import argparse
import os
from abc import abstractmethod, ABCMeta
from typing import Dict, Union, Any, List  # noqa: F401

import apache_beam as beam
import google.protobuf.text_format
import six
import tensorflow as tf  # noqa: F401
import tensorflow_transform as tft
from apache_beam.io import tfrecordio
from apache_beam.io.filesystem import CompressionTypes
from apache_beam.io.filesystems import FileSystems
from apache_beam.runners import PipelineState  # noqa: F401
from spotify_tensorflow.tfx.utils import assert_not_empty_string, assert_not_none
from tensorflow.python.lib.io import file_io
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_transform.beam import impl as beam_impl
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow_transform.coders import ExampleProtoCoder
from tensorflow_transform.tf_metadata import dataset_metadata, schema_utils
from tensorflow_transform.tf_metadata import dataset_schema


@six.add_metaclass(ABCMeta)
class TFTransform:
    """Abstract class for TFX.TFT"""

    @abstractmethod
    def get_preprocessing_fn(self):  # type: () -> Dict[str, Any]
        """Returns a users defined preprocessing function"""
        pass

    def transform_data(self,
                       dataflow_args,
                       temp_location,
                       training_data,
                       training_data_transformed_dst,
                       evaluation_data,
                       evaluation_data_transformed_dst,
                       feature_spec,
                       transform_fn_dst):
        tftransform(dataflow_args,
                    temp_location,
                    self.get_preprocessing_fn(),
                    feature_spec,
                    transform_fn_dst,
                    training_data,
                    training_data_transformed_dst,
                    evaluation_data,
                    evaluation_data_transformed_dst)

    @classmethod
    def run(cls):
        if not issubclass(cls, TFTransform):
            raise ValueError("Class {} should be inherit from TFT".format(cls))

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--training_data_src",
            required=True,
            help="path to the raw feature for training")
        parser.add_argument(
            "--evaluation_data_src",
            required=False,
            help="path to the raw feature for evaluation")
        parser.add_argument(
            "--training_data_dst",
            required=True,
            help="path to the transformed feature for training")
        parser.add_argument(
            "--evaluation_data_dst",
            required=True,
            help="path to the transformed feature for evaluation")
        parser.add_argument(
            "--schema_file",
            required=True,
            help="path to the schema txt file")
        parser.add_argument(
            "--temp_location",
            required=True,
            help="temporary location for tf.transform job")
        parser.add_argument(
            "--transform_fn_dst",
            required=False,
            help="path to the output transform function")

        tft_args, dataflow_args = parser.parse_known_args()

        # parse schema and convert it to feature spec
        schema = schema_pb2.Schema()
        schema_text = file_io.read_file_to_string(tft_args.schema_file)
        google.protobuf.text_format.Parse(schema_text, schema)
        feature_spec = schema_utils.schema_as_feature_spec(schema).feature_spec

        p = cls()
        p.transform_data(dataflow_args,
                         tft_args.temp_location,
                         tft_args.training_data_src,
                         tft_args.training_data_dst,
                         tft_args.evaluation_data_src,
                         tft_args.evaluation_data_dst,
                         feature_spec,
                         tft_args.transform_fn_dst)


def tftransform(dataflow_args,  # type: List[str]
                temp_location,  # type: str
                preprocessing_fn,  # type: Any
                feature_spec,  # type: Dict[str, Union[tf.FixedLenFeature, tf.VarLenFeature]]  # noqa: E501
                transform_fn_dst,  # type: str
                training_data,  # type: str
                training_data_transformed_dst_dir,  # type: str
                evaluation_data=None,  # type: Union[None, str]
                evaluation_data_transformed_dst_dir=None,  # type: Union[None, str]
                compression_type=CompressionTypes.AUTO  # type: str
                ):  # type: (...) -> PipelineState
    """
    Generic tf.transform pipeline that takes tf.{example, record} training and evaluation
    datasets and outputs transformed data together with transform function Saved Model.

    :param dataflow_args: un-parsed Dataflow arguments
    :param temp_location: temporary location for Dataflow
    :param preprocessing_fn: tf.transform preprocessing function
    :param feature_spec: TensorFlow feature spec
    :param transform_fn_dst: location for the transform function Saved Model
    :param training_data: path/regex of the training data
    :param training_data_transformed_dst_dir: directory for the transformed training data
    :param evaluation_data: path/regex of the evaluation data
    :param evaluation_data_transformed_dst_dir: directory for the transformed evaluation data
    :param compression_type: compression type for writing of tf.records
    :return final state of the Beam pipeline
    """
    assert_not_empty_string(temp_location)
    assert_not_none(preprocessing_fn)
    assert_not_none(feature_spec)
    assert_not_empty_string(transform_fn_dst)
    assert_not_empty_string(training_data)
    assert_not_empty_string(training_data_transformed_dst_dir)

    raw_schema = dataset_schema.from_feature_spec(feature_spec)
    raw_data_metadata = dataset_metadata.DatasetMetadata(raw_schema)

    # convert our temp-location to sth Beam can understand
    if not any(i.startswith("--temp_location=") for i in dataflow_args):
        dataflow_args.append("--temp_location={}".format(temp_location))

    if not any(i.startswith("--job_name") for i in dataflow_args):
        import getpass
        import time
        dataflow_args.append("--job_name=tf-transform-{}-{}".format(getpass.getuser(),
                                                                    int(time.time())))

    tft_model_path = os.path.join(transform_fn_dst,
                                  tft.TFTransformOutput.TRANSFORM_FN_DIR,
                                  "saved_model.pb")

    if FileSystems.exists(tft_model_path):
        raise ValueError(
            "Transform function destination {} already exists!".format(transform_fn_dst))

    pipeline = beam.Pipeline(argv=dataflow_args)
    with beam_impl.Context(temp_dir=temp_location):
        coder = ExampleProtoCoder(raw_data_metadata.schema)
        raw_data = (
                pipeline
                | "ReadTrainData" >> tfrecordio.ReadFromTFRecord(training_data)  # noqa: E501
                | "DecodeTrain" >> beam.Map(coder.decode))

        raw_dataset = (raw_data, raw_data_metadata)
        transformed_dataset, transform_fn = (
                raw_dataset
                | "AnalyzeAndTransform" >> beam_impl.AnalyzeAndTransformDataset(preprocessing_fn))

        transformed_data, transformed_metadata = transformed_dataset
        transformed_data_coder = ExampleProtoCoder(transformed_metadata.schema)

        _ = (
                transformed_data
                | "EncodeTrainData" >> beam.Map(transformed_data_coder.encode)
                | "WriteTrainData" >> tfrecordio.WriteToTFRecord(os.path.join(training_data_transformed_dst_dir, "part"),  # noqa: E501
                                                                 compression_type=compression_type,  # noqa: E501
                                                                 file_name_suffix=".tfrecords"))  # noqa: E501
        if evaluation_data is not None:
            assert evaluation_data
            assert evaluation_data_transformed_dst_dir
            raw_eval_data = (
                    pipeline
                    | "ReadEvalData" >> tfrecordio.ReadFromTFRecord(evaluation_data,
                                                                    coder=ExampleProtoCoder(raw_data_metadata.schema),  # noqa: E501
                                                                    validate=True))
            raw_eval_dataset = (raw_eval_data, raw_data_metadata)
            transformed_test_dataset = (
                    (raw_eval_dataset, transform_fn) | beam_impl.TransformDataset())
            transformed_eval_data, _ = transformed_test_dataset

            _ = (
                    transformed_eval_data
                    | "EncodeEvalData" >> beam.Map(transformed_data_coder.encode)
                    | "WriteEvalData" >> tfrecordio.WriteToTFRecord(os.path.join(evaluation_data_transformed_dst_dir, "part"),  # noqa: E501
                                                                    compression_type=compression_type,  # noqa: E501
                                                                    file_name_suffix=".tfrecords"))  # noqa: E501

        _ = (   # noqa: F841
                transform_fn
                | "WriteTransformFn" >>
                transform_fn_io.WriteTransformFn(transform_fn_dst))
    result = pipeline.run().wait_until_finish()

    return result
