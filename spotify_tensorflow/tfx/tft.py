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
import getpass
import os
import sys
import time
import warnings
from typing import Any, Union, List  # noqa: F401

import apache_beam as beam
from apache_beam.io import tfrecordio
from apache_beam.io.filesystem import CompressionTypes
from apache_beam.io.filesystems import FileSystems
from apache_beam.runners import PipelineState  # noqa: F401
from spotify_tensorflow.tf_schema_utils import schema_txt_to_feature_spec
from spotify_tensorflow.tfx.utils import assert_not_empty_string, construct_tft_reqs_txt, \
    assert_not_none
from tensorflow_transform.beam import impl as beam_impl
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow_transform.coders import ExampleProtoCoder
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema


class TFTransform(object):
    def __init__(self, preprocessing_fn):
        self.preprocessing_fn = preprocessing_fn

    def run(self, args=None):
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
        parser.add_argument(
            "--requirements_file",
            required=False,
            help="path to the requirements file we would like installed on the Dataflow workers")
        parser.add_argument(
            "--compression_type",
            required=False,
            help="compression type for writing of tf.records")

        if args is None:
            args = sys.argv[1:]
        tft_args, pipeline_args = parser.parse_known_args(args=args)

        if tft_args.requirements_file is None:
            reqs_file = construct_tft_reqs_txt()
        else:
            reqs_file = tft_args.requirements_file

        # pipeline_args also needs temp_location and requirements_file
        pipeline_args.append("--temp_location=%s" % tft_args.temp_location)
        pipeline_args.append("--requirements_file=%s" % reqs_file)

        tftransform(pipeline_args=pipeline_args,
                    temp_location=tft_args.temp_location,
                    schema_file=tft_args.schema_file,
                    output_dir=tft_args.output_dir,
                    preprocessing_fn=self.preprocessing_fn,
                    training_data=tft_args.training_data,
                    evaluation_data=tft_args.evaluation_data,
                    transform_fn_dir=tft_args.transform_fn_dir,
                    compression_type=tft_args.compression_type)


def tftransform(pipeline_args,                          # type: List[str]
                temp_location,                          # type: str
                schema_file,                            # type: str
                output_dir,                             # type: str
                preprocessing_fn,                       # type: Any
                training_data=None,                     # type: Union[None, str]
                evaluation_data=None,                   # type: Union[None, str]
                transform_fn_dir=None,                  # type: Union[None, str]
                compression_type=None                   # type: str
                ):  # type: (...) -> PipelineState
    """
    Generic tf.transform pipeline that takes tf.{example, record} training and evaluation
    datasets and outputs transformed data together with transform function Saved Model.

    :param pipeline_args: un-parsed Dataflow arguments
    :param temp_location: temporary location for dataflow job working dir
    :param schema_file: path to the raw feature schema text file
    :param output_dir: output dir for transformed data and function
    :param preprocessing_fn: tf.transform preprocessing function
    :param training_data: path to the training data
    :param evaluation_data: path to the evaluation data
    :param transform_fn_dir: dir to previously saved transformation function to apply
    :param compression_type: compression type for writing of tf.records
    :return final state of the Beam pipeline
    """
    assert_not_empty_string(temp_location)
    assert_not_empty_string(schema_file)
    assert_not_empty_string(output_dir)
    assert_not_none(preprocessing_fn)

    if compression_type is None:
        compression_type = CompressionTypes.AUTO

    raw_feature_spec = schema_txt_to_feature_spec(schema_file)
    raw_schema = dataset_schema.from_feature_spec(raw_feature_spec)
    raw_data_metadata = dataset_metadata.DatasetMetadata(raw_schema)
    raw_data_coder = ExampleProtoCoder(raw_data_metadata.schema)

    transformed_train_output_dir = os.path.join(output_dir, "training")
    transformed_eval_output_dir = os.path.join(output_dir, "evaluation")

    if not any(i.startswith("--job_name") for i in pipeline_args):
        pipeline_args.append("--job_name=tf-transform-{}-{}".format(getpass.getuser(),
                                                                    int(time.time())))

    pipeline = beam.Pipeline(argv=pipeline_args)
    with beam_impl.Context(temp_dir=temp_location):
        if training_data is not None:
            # if training data is provided, transform_fn_dir will be ignored
            if transform_fn_dir is not None:
                warnings.warn("Transform_fn_dir is ignored because training_data is provided")

            transform_fn_output = os.path.join(output_dir, "transform_fn", "saved_model.pb")
            if FileSystems.exists(transform_fn_output):
                raise ValueError("Transform fn already exists at %s!" % transform_fn_output)

            # compute the transform_fn and apply to the training data
            raw_train_data = (
                    pipeline
                    | "ReadTrainData" >> tfrecordio.ReadFromTFRecord(training_data,
                                                                     coder=raw_data_coder))

            ((transformed_train_data, transformed_train_metadata), transform_fn) = (
                    (raw_train_data, raw_data_metadata)
                    | ("AnalyzeAndTransformTrainData" >> beam_impl.AnalyzeAndTransformDataset(preprocessing_fn)))  # noqa: E501

            _ = (   # noqa: F841
                    transform_fn
                    | "WriteTransformFn" >>
                    transform_fn_io.WriteTransformFn(output_dir))

            transformed_train_coder = ExampleProtoCoder(transformed_train_metadata.schema)
            _ = (   # noqa: F841
                    transformed_train_data
                    | "WriteTransformedTrainData" >> tfrecordio.WriteToTFRecord(os.path.join(transformed_train_output_dir, "part"),  # noqa: E501
                                                                                coder=transformed_train_coder,  # noqa: E501
                                                                                compression_type=compression_type,  # noqa: E501
                                                                                file_name_suffix=".tfrecords"))  # noqa: E501
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
                                                                    coder=raw_data_coder))

            (transformed_eval_data, transformed_eval_metadata) = (
                    ((raw_eval_data, raw_data_metadata), transform_fn)
                    | "TransformEvalData" >> beam_impl.TransformDataset())

            transformed_eval_coder = ExampleProtoCoder(transformed_eval_metadata.schema)
            _ = (   # noqa: F841
                    transformed_eval_data
                    | "WriteTransformedEvalData" >> tfrecordio.WriteToTFRecord(os.path.join(transformed_eval_output_dir, "part"),  # noqa: E501
                                                                               coder=transformed_eval_coder,  # noqa: E501
                                                                               compression_type=compression_type,  # noqa: E501
                                                                               file_name_suffix=".tfrecords"))  # noqa: E501
    result = pipeline.run().wait_until_finish()

    return result
