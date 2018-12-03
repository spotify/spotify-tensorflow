# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from abc import abstractmethod, ABCMeta
from typing import Tuple, Dict, Union, Any, List  # noqa: F401

import apache_beam as beam
import numpy as np
import six
import tensorflow as tf
import tensorflow_transform as tft
from apache_beam.io import tfrecordio
from apache_beam.io.filesystem import CompressionTypes
from apache_beam.io.filesystems import FileSystems
from apache_beam.runners import PipelineState
from tensorflow_metadata.proto.v0.schema_pb2 import INT, FLOAT, BYTES
from tensorflow_metadata.proto.v0.schema_pb2 import Schema
from tensorflow_transform.beam import impl as beam_impl
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow_transform.coders import ExampleProtoCoder
from tensorflow_transform.tf_metadata import dataset_metadata
from spotify_tensorflow.flags import validate_required_flags


@six.add_metaclass(ABCMeta)
class TFT:
    """Abstract class for TFX.TFT"""

    def __init__(self, schema_path):
        # type: (str) -> None
        assert schema_path
        self.schema_path = schema_path

    @property
    def feature_spec(self):
        """Returns raw feature spec."""
        try:
            return self.__feature_spec_lazy_val
        except AttributeError:
            self.__feature_spec_lazy_val = parse_tf_metadata_schema(self.schema_path)
            return self.__feature_spec_lazy_val

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
                       transform_fn_dst):
        tftransform(dataflow_args,
                    temp_location,
                    self.get_preprocessing_fn(),
                    self.feature_spec,
                    transform_fn_dst,
                    training_data,
                    training_data_transformed_dst,
                    evaluation_data,
                    evaluation_data_transformed_dst,
                    self.schema_path)

    @classmethod
    def get_std_main(cls):
        if not issubclass(cls, TFT):
            raise ValueError("Class {} should be inherit from TFT".format(cls))

        def inner_tf_compatible_main(dataflow_args):
            validate_required_flags()
            p = cls(tf.flags.FLAGS.schema_file)
            return p.transform_data(
                dataflow_args,
                tf.flags.FLAGS.temp_location,
                tf.flags.FLAGS.training_data_src,
                tf.flags.FLAGS.training_data_dst,
                tf.flags.FLAGS.evaluation_data_src,
                tf.flags.FLAGS.evaluation_data_dst,
                tf.flags.FLAGS.transform_fn_dst)
        return inner_tf_compatible_main


__to_tf_type = {INT: tf.int32,
                FLOAT: tf.float32,
                BYTES: tf.string}


__to_np_type = {INT: np.int32,
                FLOAT: np.float32,
                BYTES: np.bytes_}


def __assert_not_none(arg):
    # type: (Any) -> None
    if arg is None:
        raise TypeError("Argument can't be a None")


def __assert_not_empty_string(arg):
    # type: (Any) -> None
    if not isinstance(arg, str):
        raise TypeError("Argument should be a string")
    if arg == "":
        raise ValueError("Argument can't be an empty string")


def tf_metadata_schema_feature_to_feature_spec(feature):
    # type: (Any) -> Tuple[str, Union[tf.FixedLenFeature, tf.VarLenFeature]]
    """Maps tf.metadata.Schema Feature to TF Feature spec.
    :param feature: tf.metadata.Schema Feature
    :return: a tuple of feature name and TF feature (FixedLenFeature or VarLenFeature).
    """
    # for PoC cover only fixed/var len features
    __assert_not_none(feature)
    if feature.HasField("shape"):
        shape = tuple(d.size for d in feature.shape.dim)
        # NOTE: we set default_value only for the PoC, we would not do this in production!
        default_value = np.zeros(shape, __to_np_type[feature.type])
        return feature.name, tf.FixedLenFeature(shape=shape,
                                                dtype=__to_tf_type[feature.type],
                                                default_value=default_value)
    else:
        return feature.name, tf.VarLenFeature(dtype=__to_tf_type[feature.type])


def parse_tf_metadata_schema(schema_path):
    # type: (str) -> Dict[str, Union[tf.FixedLenFeature, tf.VarLenFeature]]
    """Parses tf.metadata schema as TF feature spec.
    :param schema_path: path to protobuf file of serialized tf.metadata Schema.
    :return: TF feature spec.
    """
    __assert_not_empty_string(schema_path)
    schema = Schema()
    with FileSystems.open(schema_path) as f:
        schema.ParseFromString(f.read())

    return dict(tf_metadata_schema_feature_to_feature_spec(f) for f in schema.feature)


def get_raw_schema_path_from_extra_assets(transform_fn_dst):  # type: (str) -> str
    """
    Returns path to the raw feature schema of the tf.transform transform function's Saved Model
    assets.
    :param transform_fn_dst: tf.transform transform function destination path
    """
    __assert_not_empty_string(transform_fn_dst)
    return os.path.join(transform_fn_dst,
                        tft.TFTransformOutput.TRANSFORM_FN_DIR,
                        "assets.extra",
                        "raw_schema.pb")


def tftransform(dataflow_args,  # type: List[str]
                temp_location,  # type: str
                preprocessing_fn,  # type: Any
                feature_spec,  # type: Dict[str, Union[tf.FixedLenFeature, tf.VarLenFeature]]  # noqa: E501
                transform_fn_dst,  # type: str
                training_data,  # type: str
                training_data_transformed_dst_dir,  # type: str
                evaluation_data=None,  # type: Union[None, str]
                evaluation_data_transformed_dst_dir=None,  # type: Union[None, str]
                schema_path=None,  # type: Union[None, str]
                compression_type=CompressionTypes.AUTO  # type: str
                ):  # type: (...) -> PipelineState
    """
    Generic tf.transform pipeline that takes tf.{example, record} training and evaluation
    datasets and outputs transformed data together with transform function Saved Model.

    :param dataflow_args: un-parsed Dataflow arguments
    :param temp_location: temporary location for Dataflow
    :param preprocessing_fn: tf.transform preprocessing function
    :param feature_spec: TensorFlow feature spec
    :param training_data: path/regex of the training data
    :param training_data_transformed_dst_dir: directory for the transformed training data
    :param evaluation_data: path/regex of the evaluation data
    :param evaluation_data_transformed_dst_dir: directory for the transformed evaluation data
    :param transform_fn_dst: location for the transform function Saved Model
    :param schema_path: Path to tf.metadata Schema protobuf file
    :param compression_type: compression type for writing of tf.records
    :return final state of the Beam pipeline
    """
    __assert_not_empty_string(temp_location)
    __assert_not_none(preprocessing_fn)
    __assert_not_none(feature_spec)
    __assert_not_empty_string(transform_fn_dst)
    __assert_not_empty_string(training_data)
    __assert_not_empty_string(training_data_transformed_dst_dir)

    raw_data_metadata = dataset_metadata.DatasetMetadata(
        dataset_metadata.dataset_schema.from_feature_spec(feature_spec))

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
        # fail here before the job submission and failure at job runtime
        raise ValueError(
            "Transform function destination {} already exists!".format(transform_fn_dst))

    pipeline = beam.Pipeline(argv=dataflow_args)
    with beam_impl.Context(temp_dir=temp_location):
        raw_data = (
                pipeline
                | "ReadTrainData" >> tfrecordio.ReadFromTFRecord(training_data,
                                                                 coder=ExampleProtoCoder(raw_data_metadata.schema),  # noqa: E501
                                                                 validate=True))

        raw_dataset = (raw_data, raw_data_metadata)
        transformed_dataset, transform_fn = (
                raw_dataset
                | beam_impl.AnalyzeAndTransformDataset(preprocessing_fn))

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

    if result == PipelineState.DONE:
        raw_schema = get_raw_schema_path_from_extra_assets(transform_fn_dst)
        FileSystems.copy([schema_path], [raw_schema])

    return result
