#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2019 Spotify AB
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis.post_export_metrics import post_export_metrics
from tensorflow_model_analysis.post_export_metrics.metric_keys import base_key

NDCG_AT_K = base_key("ndcg_at_k")


# Similar to https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/post_export_metrics/post_export_metrics.py#L43  # noqa
def ndcg_at_k(*args, **kwargs):
    """This is the function that the user calls."""

    def callback(features_dict, predictions_dict, labels_dict):
        """This actual callback that goes into add_metrics_callbacks."""
        metric = _NdcgAtK(*args, **kwargs)
        metric.check_compatibility(features_dict, predictions_dict, labels_dict)
        metric_ops = {}
        for key, value in metric.get_metric_ops(
            features_dict, predictions_dict, labels_dict
        ).items():
            metric_ops[key] = value
        return metric_ops

    # We store the metric's export name in the .name property of the callback.
    callback.name = "ndcg_at_k"
    callback.populate_stats_and_pop = _NdcgAtK(*args, **kwargs).populate_stats_and_pop
    return callback


# Similar to https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/post_export_metrics/post_export_metrics.py#L1375  # noqa
class _NdcgAtK(post_export_metrics._PrecisionRecallAtK):
    """Metric that computes the NDCG at K for classification models.

      Create a metric that computes NDCG at K.

      Predictions should be a dict containing the CLASSES key and PROBABILITIES
      keys. Predictions should have the same size for all examples. The model
      should NOT, for instance, produce 2 classes on one example and 4 classes on
      another example.

      Labels should be a string Tensor, or a SparseTensor whose dense form is
      a string Tensor whose entries are the corresponding labels. Note that the
      values of the CLASSES in the predictions and that of labels will be compared
      directly, so they should come from the "same vocabulary", so if predictions
      are class IDs, then labels should be class IDs, and so on.
      """

    def __init__(
        self,
        cutoffs,
        example_weight_key=None,
        target_prediction_keys=None,
        labels_key=None,
        metric_tag=None,
        classes_key=None,
        probabilities_key=None,
    ):
        """Creates a metric that computes the ndcg at `k`.

        Args:
          cutoffs: List of `k` values at which to compute the ndcg.
            Use a value of `k` = 0 to indicate that all predictions should be
            considered.
          example_weight_key: The optional key of the example weight column in the
            features_dict. If not given, all examples will be assumed to have a
            weight of 1.0.
          target_prediction_keys: Optional acceptable keys in predictions_dict in
            descending order of precedence.
          labels_key: Optionally, the key from labels_dict to use.
          metric_tag: If provided, a custom metric tag. Only necessary to
            disambiguate instances of the same metric on different predictions.
          classes_key: Optionally, the key from predictions that specifies classes.
          probabilities_key: Optionally, the key from predictions that specifies
            probabilities.
        """
        super(_NdcgAtK, self).__init__(
            NDCG_AT_K,
            cutoffs,
            example_weight_key,
            target_prediction_keys,
            labels_key,
            metric_tag,
            classes_key,
            probabilities_key,
        )

    # Override
    def get_metric_ops(self, features_dict, predictions_dict, labels_dict):

        squeezed_weights = None
        if self._example_weight_key:
            squeezed_weights = post_export_metrics._flatten_to_one_dim(
                features_dict[self._example_weight_key]
            )

        labels = labels_dict
        if self._labels_key:
            labels = labels_dict[self._labels_key]
        if isinstance(labels_dict, tf.SparseTensor):
            labels = tf.sparse_tensor_to_dense(labels_dict, default_value="")

        # Expand dims if necessary.
        labels = tf.cond(
            tf.equal(tf.rank(labels), 1), lambda: tf.expand_dims(labels, -1), lambda: labels
        )

        classes = post_export_metrics._get_target_tensor(predictions_dict, self._classes_keys)
        scores = post_export_metrics._get_target_tensor(predictions_dict, self._probabilities_keys)

        # To support canned Estimators which right now only expose the argmax class
        # id, if labels are ints then then the classes are likely class_ids in
        # string form, so we can automatically expand the classes to the full set
        # for matching the labels (see b/113170729).
        # Not tested
        if labels.dtype == tf.int64:
            classes = tf.cond(
                # Match only when classes has a single item (i.e. argmax).
                tf.equal(tf.shape(classes)[-1], 1),
                lambda: tf.as_string(post_export_metrics._class_ids(scores)),
                lambda: classes,
            )

        labels = post_export_metrics._cast_or_convert(labels, classes.dtype)

        metric_ops = _ndcg_at_k_metric(classes, scores, labels, self._cutoffs, squeezed_weights)
        return {self._metric_key(self._metric_name): metric_ops}


# Similar to https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/post_export_metrics/metrics.py#L224  # noqa
def _ndcg_at_k_metric(classes, scores, labels, cutoffs, weights=None):
    # pyformat: disable
    """NDCG at `k`.

    Args:
      classes: Tensor containing class names. Should be a BATCH_SIZE x NUM_CLASSES
        Tensor.
      scores: Tensor containing the associated scores. Should be a
        BATCH_SIZE x NUM_CLASSES Tensor.
      labels: Tensor containing the true labels. Should be a rank-2 Tensor where
        the first dimension is BATCH_SIZE. The second dimension can be anything.
      cutoffs: List containing the values for the `k` at which to compute the
        ndcg for. Use a value of `k` = 0 to indicate that all predictions
        should be considered.
      weights: Optional weights for each of the examples. If None,
        each of the predictions/labels will be assumed to have a weight of 1.0.
        If present, should be a BATCH_SIZE Tensor.

    The value_op will return a matrix with len(cutoffs) rows and 2 columns:
    [ cutoff 0, ndcg at cutoff 0 ]
    [ cutoff 1, ndcg at cutoff 1 ]
    [     :                :          ]
    [ cutoff n, ndcg at cutoff n ]

    Returns:
      (value_op, update_op) for the NDCG at K metric.
    """
    # pyformat: enable
    num_cutoffs = len(cutoffs)

    scope = "ndcg_at_k"

    with tf.variable_scope(scope, [classes, scores, labels]):

        # Predicted positive.
        ndcgs_at_k = tf.Variable(
            initial_value=[0.0] * num_cutoffs,
            dtype=tf.float64,
            trainable=False,
            collections=[tf.GraphKeys.METRIC_VARIABLES, tf.GraphKeys.LOCAL_VARIABLES],
            validate_shape=True,
            name="ndcgs_at_k",
        )

        # Predicted positive.
        total_weights = tf.Variable(
            initial_value=0.0,
            dtype=tf.float64,
            trainable=False,
            collections=[tf.GraphKeys.METRIC_VARIABLES, tf.GraphKeys.LOCAL_VARIABLES],
            validate_shape=True,
            name="total_weights",
        )
        if weights is not None:
            weights_f64 = tf.cast(weights, tf.float64)
        else:
            weights_f64 = tf.ones(tf.shape(labels)[0], tf.float64)

    # Value op returns
    # [ K | ndcg at K ]
    # PyType doesn't like TF operator overloads: b/92797687
    # pytype: disable=unsupported-operands
    ndcg_op = ndcgs_at_k / total_weights
    # pytype: enable=unsupported-operands
    value_op = tf.transpose(tf.stack([cutoffs, ndcg_op], axis=0))

    ndcgs_at_k_update, total_weights_update = tf.py_func(
        compute_batch_updates,
        [classes, scores, labels, weights_f64, cutoffs],
        [tf.float64, tf.float64],
    )

    update_op = tf.group(
        tf.assign_add(ndcgs_at_k, ndcgs_at_k_update),
        tf.assign_add(total_weights, total_weights_update),
    )

    return value_op, update_op


def compute_batch_updates(classes, scores, labels, weights, cutoffs):
    """Compute ndcg intermediate stats for a batch.
    This was originally defined in the body of _ndcg_at_k_metric

    Args:
      classes: Tensor containing class names. Should be a BATCH_SIZE x
        NUM_CLASSES Tensor.
      scores: Tensor containing the associated scores. Should be a BATCH_SIZE x
        NUM_CLASSES Tensor.
      labels: Tensor containing the true labels. Should be a rank-2 Tensor where
        the first dimension is BATCH_SIZE. The second dimension can be anything.
      weights: Weights for the associated exmaples. Should be a BATCH_SIZE
        Tensor.
      cutoffs: List of values to limit predictions

    Returns:
      NDCGs at K computed for the
      batch of examples.

    Raises:
      ValueError: classes and scores have different shapes; or labels has
       a different batch size from classes and scores
    """

    if classes.shape != scores.shape:
        raise ValueError(
            "classes and scores should have same shape, but got "
            "%s and %s" % (classes.shape, scores.shape)
        )

    batch_size = classes.shape[0]
    num_classes = classes.shape[1]

    if any([cutoff > num_classes for cutoff in cutoffs]):
        raise ValueError(
            "There is %d classes in total. It does not make sense to use cutoffs of %d"
            % (num_classes, cutoffs)
        )
    if labels.shape[0] != batch_size:
        raise ValueError(
            "labels should have the same batch size of %d, but got "
            "%d instead" % (batch_size, labels.shape[0])
        )

    # Sort classes, by row, by their associated scores, in descending order of score.
    sorted_classes = np.flip(classes[np.arange(batch_size)[:, None], np.argsort(scores)], axis=1)

    ndcgs = np.zeros(len(cutoffs), dtype=np.float64)
    total_weights = 0.0

    for predicted_row, label_row, weight in zip(sorted_classes, labels, weights):

        label_set = set(label_row)
        label_set.discard(b"")  # Remove filler elements.

        for i, cutoff in enumerate(cutoffs):
            cutoff_to_use = cutoff if cutoff > 0 else num_classes
            cut_predicted_row = predicted_row[:cutoff_to_use]
            # Careful here: we are not computing the ideal dcg at K but the ideal dcg at
            # K that can be obtained from these labels:
            # If the len(labels) < k, then we compute idcg(len(labels)) not idck(k)
            weighted_idcg_at_k = dcg(label_row[:cutoff_to_use], label_set) * weight
            ndcgs[i] += dcg(cut_predicted_row, label_set) / weighted_idcg_at_k

        total_weights += weight

    return ndcgs, total_weights


def dcg(retrieved_elements, relevant_elements):
    if len(retrieved_elements) == 0 or len(relevant_elements) == 0:
        raise ValueError("Either of retrieved_elements or relevant_elements was empty")
    # Computes an ordered vector of 1.0 and 0.0
    score = np.array([float(el in relevant_elements) for el in retrieved_elements])
    return np.sum(score / np.log2(1 + np.arange(1, len(score) + 1)))
