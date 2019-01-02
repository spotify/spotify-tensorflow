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
import six
from apache_beam.io import tfrecordio
from apache_beam.io.filesystem import CompressionTypes
from apache_beam.runners import PipelineState  # noqa: F401
from spotify_tensorflow.tf_schema_utils import schema_txt_to_feature_spec
from spotify_tensorflow.tfx.utils import assert_not_empty_string, assert_not_none
from tensorflow_transform.beam import impl as beam_impl
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow_transform.coders import ExampleProtoCoder
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema


@six.add_metaclass(ABCMeta)
class TFTransform:

    """Abstract class for TFX.TFT"""

    @abstractmethod
    def get_preprocessing_fn(self):  # type: () -> Dict[str, Any]
        """Returns a users defined preprocessing function"""
        pass

    def transform_data(self,
                       pipeline_args,
                       temp_location,
                       schema_file,
                       output_dir,
                       training_data,
                       evaluation_data,
                       transform_fn_dir):
        tftransform(pipeline_args=pipeline_args,
                    temp_location=temp_location,
                    preprocessing_fn=self.get_preprocessing_fn(),
                    schema_file=schema_file,
                    output_dir=output_dir,
                    training_data=training_data,
                    evaluation_data=evaluation_data,
                    transform_fn_dir=transform_fn_dir)

    @classmethod
    def run(cls):
        if not issubclass(cls, TFTransform):
            raise ValueError("Class {} should be inherit from TFT".format(cls))

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--training_data",
            required=False,
            help="path to the raw feature for training")
        parser.add_argument(
            "--evaluation_data",
            required=False,
            help="path to the raw feature for evaluation")
        parser.add_argument(
            "--output_dir",
            required=True,
            help="output dir for data transformation")
        parser.add_argument(
            "--schema_file",
            required=True,
            help="path to the schema txt file")
        parser.add_argument(
            "--temp_location",
            required=True,
            help="temporary working dir for tf.transform job")
        parser.add_argument(
            "--transform_fn_dir",
            required=False,
            help="path to the saved transform function")

        tft_args, pipeline_args = parser.parse_known_args()

        p = cls()
        p.transform_data(pipeline_args=pipeline_args,
                         temp_location=tft_args.temp_location,
                         schema_file=tft_args.schema_file,
                         output_dir=tft_args.output_dir,
                         training_data=tft_args.training_data,
                         evaluation_data=tft_args.evaluation_data,
                         transform_fn_dir=tft_args.transform_fn_dir)


def tftransform(pipeline_args,                          # type: List[str]
                temp_location,                          # type: str
                preprocessing_fn,                       # type: Any
                schema_file,                            # type: str
                output_dir,                             # type: str
                training_data=None,                     # type: Union[None, str]
                evaluation_data=None,                   # type: Union[None, str]
                transform_fn_dir=None,                  # type: Union[None, str]
                compression_type=CompressionTypes.AUTO  # type: str
                ):  # type: (...) -> PipelineState
    """
    Generic tf.transform pipeline that takes tf.{example, record} training and evaluation
    datasets and outputs transformed data together with transform function Saved Model.

    :param pipeline_args: un-parsed Dataflow arguments
    :param temp_location: temporary location for dataflow job working dir
    :param preprocessing_fn: tf.transform preprocessing function
    :param schema_file: path to the raw feature schema text file
    :param output_dir: output dir for transformed data and function
    :param training_data: path to the training data
    :param evaluation_data: path to the evaluation data
    :param transform_fn_dir: dir to previously saved transformation function to apply
    :param compression_type: compression type for writing of tf.records
    :return final state of the Beam pipeline
    """
    assert_not_empty_string(temp_location)
    assert_not_none(preprocessing_fn)
    assert_not_empty_string(schema_file)
    assert_not_empty_string(output_dir)

    feature_spec = schema_txt_to_feature_spec(schema_file)
    raw_schema = dataset_schema.from_feature_spec(feature_spec)
    raw_data_metadata = dataset_metadata.DatasetMetadata(raw_schema)
    coder = ExampleProtoCoder(raw_data_metadata.schema)

    transformed_train_output_dir = os.path.join(output_dir, "training")
    transformed_eval_output_dir = os.path.join(output_dir, "evaluation")
    transformed_fn_output_dir = os.path.join(output_dir, "transform_fn")

    if not any(i.startswith("--job_name") for i in pipeline_args):
        import getpass
        import time
        pipeline_args.append("--job_name=tf-transform-{}-{}".format(getpass.getuser(),
                                                                    int(time.time())))

    pipeline = beam.Pipeline(argv=pipeline_args)
    with beam_impl.Context(temp_dir=temp_location):
        if training_data is not None:
            # if training data is provided, transform_fn_dir will be ignored
            if transform_fn_dir is not None:
                import warnings
                warnings.warn("Transform_fn_dir will be ignored since training_data is provided")

            # compute the transform_fn and apply to the training data
            raw_train_data = (
                    pipeline
                    | "ReadTrainData" >> tfrecordio.ReadFromTFRecord(training_data)
                    | "DecodeTrain" >> beam.Map(coder.decode))

            ((transformed_train_data, transformed_train_metadata), transform_fn) = (
                    (raw_train_data, raw_data_metadata)
                    | ("AnalyzeAndTransform" >> beam_impl.AnalyzeAndTransformDataset(preprocessing_fn)))  # noqa: E501

            _ = (   # noqa: F841
                    transform_fn
                    | "WriteTransformFn" >>
                    transform_fn_io.WriteTransformFn(transformed_fn_output_dir))

            transformed_train_coder = ExampleProtoCoder(transformed_train_metadata.schema)
            _ = (   # noqa: F841
                    transformed_train_data
                    | "EncodeTrainData" >> beam.Map(transformed_train_coder.encode)
                    | "WriteTrainData" >> tfrecordio.WriteToTFRecord(os.path.join(transformed_train_output_dir, "part"),  # noqa: E501
                                                                     compression_type=compression_type,  # noqa: E501
                                                                     file_name_suffix=".tfrecords"))
        else:
            if transform_fn_dir is None:
                raise ValueError("Either training_data or transformed_fn needs to be provided")
            # load the transform_fn
            transform_fn = pipeline | transform_fn_io.ReadTransformFn(transform_fn_dir)

        if evaluation_data is not None:
            # if evaluation_data exists, apply the transform_fn to the evaluation data
            raw_eval_data = (
                    pipeline
                    | "ReadEvalData" >> tfrecordio.ReadFromTFRecord(evaluation_data,
                                                                    coder=ExampleProtoCoder(raw_data_metadata.schema),  # noqa: E501
                                                                    validate=True))
            (transformed_eval_data, transformed_eval_metadata) = (
                    ((raw_eval_data, raw_data_metadata), transform_fn)
                    | beam_impl.TransformDataset())

            transformed_eval_coder = ExampleProtoCoder(transformed_eval_metadata.schema)
            _ = (   # noqa: F841
                    transformed_eval_data
                    | "EncodeEvalData" >> beam.Map(transformed_eval_coder.encode)
                    | "WriteEvalData" >> tfrecordio.WriteToTFRecord(os.path.join(transformed_eval_output_dir, "part"),  # noqa: E501
                                                                    compression_type=compression_type,  # noqa: E501
                                                                    file_name_suffix=".tfrecords"))  # noqa: E501
    result = pipeline.run().wait_until_finish()

    return result
