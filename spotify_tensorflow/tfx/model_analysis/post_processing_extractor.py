#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2019 Spotify AB
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import apache_beam as beam
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.extractors import extractor
from spotify_tensorflow.tfx.model_analysis import fpl_utils

POST_PROCESSING_EXTRACTOR_STAGE_NAME = "PostProcessFPL"


def _Apply_PostProcessing(extracts, post_processing_fn):  # pylint: disable=invalid-name
    """Augments FPL dict with new feature(s)."""
    # Create a new feature from existing ones.
    fpl_copy = fpl_utils.get_fpl_copy_from_extracts(extracts)
    new_fpl = post_processing_fn(fpl_copy)
    if not isinstance(new_fpl, types.FeaturesPredictionsLabels):
        raise TypeError(
            "Function %s did not return a valid FeaturesPredictionsLabels." % post_processing_fn
        )

    result = copy.copy(extracts)
    result[constants.FEATURES_PREDICTIONS_LABELS_KEY] = new_fpl
    return result


@beam.ptransform_fn
@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(types.Extracts)
def _PostProcessing(extracts, post_processing_fn):  # pylint: disable=invalid-name
    """Extracts meta-features derived from existing features.

    It must be the case that the PredictExtractor was called before calling this
    function.

    Args:
      extracts: PCollection containing the Extracts that will have
        MaterializedColumn added to its extracts.
      post_processing_fn: A function that adds new features. Must take a
        FeaturesPredictionsLabel tuple as an argument, and return a a dict of new
        features to add, where the keys are new feature names and the values are
        the associated values.Only adding new features is permitted to prevent
        inadvertently removing useful data.

    Returns:
      PCollection of Extracts
    """
    return extracts | "_PostProcessing" >> beam.Map(_Apply_PostProcessing, post_processing_fn)


def PostProcessingExtractor(post_processing_fn, stage_name=POST_PROCESSING_EXTRACTOR_STAGE_NAME):
    # pylint: disable=no-value-for-parameter
    return extractor.Extractor(
        stage_name=stage_name, ptransform=_PostProcessing(post_processing_fn=post_processing_fn)
    )
