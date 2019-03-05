# -*- coding: utf-8 -*-
#
# Copyright 2017-2019 Spotify AB.
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

import argparse
import os

import apache_beam as beam
from apache_beam.io import Read, tfrecordio
from apache_beam.io.filebasedsource import FileBasedSource
from apache_beam.io.filesystem import CompressionTypes
from apache_beam.options.pipeline_options import PipelineOptions


class ExpandFileNames(beam.PTransform):
    def __init__(self, file_pattern, **kwargs):
        super(ExpandFileNames, self).__init__(**kwargs)
        self._source = FileNamesListSource(file_pattern)

    def expand(self, pvalue):
        return pvalue.pipeline | Read(self._source)


class FileNamesListSource(FileBasedSource):
    """A File source for listing files from a pattern"""

    def __init__(self, file_pattern):
        """Initialize a TFRecordSource.  See ReadFromTFRecord for details."""

        super(FileNamesListSource, self).__init__(
            file_pattern=file_pattern,
            compression_type=CompressionTypes.AUTO,
            splittable=False,
            validate=True)

    def read_records(self, file_name, offset_range_tracker):
        yield file_name


class EmitAsBatchDoFn(beam.DoFn):
    """A DoFn that buffers the records and emits them batch by batch."""

    def __init__(self, batch_size):
        self._batch_size = batch_size
        self._cached = []

    def process(self, element):
        self._cached.append(element)
        if len(self._cached) >= self._batch_size:
            emit = self._cached
            self._cached = []
            yield emit

    def finish_bundle(self, context=None):
        from apache_beam.transforms import window
        from apache_beam.utils.windowed_value import WindowedValue
        if len(self._cached) > 0:
            yield WindowedValue(self._cached, -1, [window.GlobalWindow()])


class PredictDoFn(beam.DoFn):
    """A DoFn that performs predictions with given trained model."""

    def __init__(self, model_export_dir):
        self._model_export_dir = model_export_dir

    def start_bundle(self):
        from tensorflow.contrib import predictor
        import tensorflow_transform  # noqa F401

        # We need to import the tensorflow_transform library in order to
        # register all of the ops that might be used by a saved model that
        # incorporates TFT transformations.

        self._predict_fn = predictor.from_saved_model(self._model_export_dir)

    def process(self, batch):
        return_dict = self._predict_fn({"inputs": batch})

        predictions = []
        for serialized_example, scores in zip(batch, return_dict["scores"]):
            input_example = self.deserialize_example(serialized_example)
            prediction = dict(trip_id=input_example["trip_id"],
                              scores=scores)
            predictions.append(prediction)

        yield predictions

    def deserialize_example(self, serialized_example_proto):
        import tensorflow as tf
        example = tf.train.Example()
        example.ParseFromString(serialized_example_proto)
        return example.features.feature


def make_string(predictions):
    for prediction in predictions:
        yield "%s,%.7f,%.7f" % (prediction["trip_id"], prediction["scores"][0],
                                prediction["scores"][1])


def predict_tfrecords(
        input_data_dir,
        output_dir,
        model_path,
        batch_size,
        pipeline_args,
        runner="DataflowRunner",
        compression_type=None
):
    """
    :param input_data_dir: path to the data to predict as tfrecords
    :param output_dir: output folder
    :param batch_size: batch size when running prediction.
    :param pipeline_args: un-parsed Dataflow arguments
    :param mode: Run in cloud or local
    :return final state of the Beam pipeline
    """

    if compression_type is None:
        compression_type = CompressionTypes.AUTO

    pipeline_options = PipelineOptions(flags=pipeline_args)

    with beam.Pipeline(runner, options=pipeline_options) as pipeline:
        (pipeline
         | "Read Data" >> tfrecordio.ReadFromTFRecord(input_data_dir,
                                                      compression_type=compression_type)
         | "Make a Batch" >> beam.ParDo(EmitAsBatchDoFn(batch_size))
         | "Predict" >> beam.ParDo(PredictDoFn(model_path))
         | "Make Strings" >> beam.FlatMap(make_string)
         | "Write in CSV" >> beam.io.WriteToText(
                    os.path.join(output_dir, "part"),
                    file_name_suffix=".csv",
                    header="trip_id,class_0,class_1"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-data",
        required=True,
        help="path to the data to predict as tfrecords")
    parser.add_argument(
        "--output",
        required=True,
        help="output folder")
    parser.add_argument(
        "--model-path",
        required=True,
        help="path to the model")
    parser.add_argument("--batch-size",
                        type=int,
                        default=32,
                        help="Batch size used in prediction.")
    parser.add_argument("--runner",
                        choices=["DirectRunner", "DataflowRunner"],
                        default="DataflowRunner",
                        help="whether to run the job locally or "
                             "in Cloud Dataflow.")
    parser.add_argument("--compression_type",
                        required=False,
                        help="compression type for reading of tf.records")

    known_args, pipeline_args = parser.parse_known_args()
    predict_tfrecords(
        known_args.input_data,
        known_args.output,
        known_args.model_path,
        known_args.batch_size,
        pipeline_args,
        known_args.runner,
        known_args.compression_type
    )
