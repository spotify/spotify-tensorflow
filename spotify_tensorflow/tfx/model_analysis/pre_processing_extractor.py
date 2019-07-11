#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2019 Spotify AB
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import apache_beam as beam
import tensorflow as tf
import tensorflow_model_analysis as tfma
from tensorflow_model_analysis.extractors import extractor

PRE_PROCESSING_EXTRACTOR_STAGE_NAME = "PreProcessExamples"


def _Apply_PreProcessing(extract, pre_processing_fn):  # pylint: disable=invalid-name
    """Modify tf.Examples before prediction
    Extract is a dictionary with one key tfma.constants.INPUT_KEY to a serialized tf.Example
    """
    serialized_example = copy.copy(extract[tfma.constants.INPUT_KEY])
    parsed_example = tf.train.Example.FromString(serialized_example)
    new_example = pre_processing_fn(parsed_example)
    return {tfma.constants.INPUT_KEY: new_example.SerializeToString()}


@beam.ptransform_fn
@beam.typehints.with_input_types(tfma.Extracts)
@beam.typehints.with_output_types(tfma.Extracts)
def _PreProcessing(extracts, pre_processing_fn):  # pylint: disable=invalid-name
    """Modify the serialized tf.Examples in the Extracts

    It must be the case that the PredictExtractor was NOT called before calling this
    function.

    Args:
      extracts: PCollection containing the Extracts.
      pre_processing_fn: A function that adds new features. Must take a
        FeaturesPredictionsLabel tuple as an argument, and return a a dict of new
        features to add, where the keys are new feature names and the values are
        the associated values.Only adding new features is permitted to prevent
        inadvertently removing useful data.

    Returns:
      PCollection of Extracts
    """
    return extracts | "_PreProcessing" >> beam.Map(_Apply_PreProcessing, pre_processing_fn)


def PreProcessingExtractor(pre_processing_fn, stage_name=PRE_PROCESSING_EXTRACTOR_STAGE_NAME):
    # pylint: disable=no-value-for-parameter
    return extractor.Extractor(
        stage_name=stage_name, ptransform=_PreProcessing(pre_processing_fn=pre_processing_fn)
    )
