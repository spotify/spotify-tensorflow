# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any  # noqa: F401

import six
import tensorflow as tf
from absl.flags import _argument_parser


class RequirableFlag(tf.flags.Flag):
    """Adds required constrain on the TensorFlow flag."""
    def __init__(self,
                 name,  # type: str
                 default,  # type: Any
                 help,  # type: str
                 short_name=None,  # type: str
                 required=False,  # type: bool
                 parser=_argument_parser.ArgumentParser(),  # type: _argument_parser.ArgumentParser
                 **args  # pylint: disable=redefined-builtin
                 ):
        super(RequirableFlag, self).__init__(
            parser, None, name, default, help, short_name, False, **args)
        self.required = required

    def validate_required(self):
        if self.required and self.value is None:
            raise ValueError("{} is required and must be set!".format(self.name))


class StrFlag(RequirableFlag):
    """String flavour of RequirableFlag"""
    def __init__(self,
                 name,  # type: str
                 default,  # type: Any
                 help,  # type: str
                 short_name=None,  # type: str
                 required=False,  # type: bool
                 **args  # pylint: disable=redefined-builtin
                 ):
        p = _argument_parser.ArgumentParser()
        super(StrFlag, self).__init__(name, default, help, short_name, required, p, **args)


class IntFlag(RequirableFlag):
    """Int flavour of RequirableFlag"""
    def __init__(self,
                 name,  # type: str
                 default,  # type: Any
                 help,  # type: str
                 short_name=None,  # type: str
                 required=False,  # type: bool
                 **args  # pylint: disable=redefined-builtin
                 ):
        p = _argument_parser.IntegerParser()
        super(IntFlag, self).__init__(name, default, help, short_name, required, p, **args)


class FloatFlag(tf.flags.Flag):
    """Float flavour of RequirableFlag"""
    def __init__(self,
                 name,  # type: str
                 default,  # type: Any
                 help,  # type: str
                 short_name=None,  # type: str
                 required=False,  # type: bool
                 **args  # pylint: disable=redefined-builtin
                 ):
        p = _argument_parser.FloatParser()
        super(FloatFlag, self).__init__(name, default, help, short_name, required, p, **args)


def validate_required_flags(flags=None):
    """Validate flags - should be called after parsing."""
    if flags is None:
        flags = tf.flags
    for n, _ in six.iteritems(flags.FLAGS.flag_values_dict()):
        if isinstance(flags.FLAGS[n], RequirableFlag):
            flags.FLAGS[n].validate_required()


def define_required_str_flag(name, short_name=None, help=None, flags=None, **args):
    """Define a TensorFlow string flag"""
    if flags is None:
        flags = tf.flags
    flags.DEFINE_flag(StrFlag(name,
                              None,
                              help,
                              short_name,
                              True,
                              **args))


def define_standard_tf_model_flags(flags=None):
    """Register standard TensorFlow model command line flags"""
    if flags is None:
        flags = tf.flags
    define_required_str_flag("training_data", "td", "paths of the training data - can be regex",
                             flags)
    define_required_str_flag("evaluation_data", "ed", "paths of the evaluation data - can be regex",
                             flags)
    define_required_str_flag("transform_fn", "tf",
                             "path of the TensorFlow transform function graph", flags)
    define_required_str_flag("job_dir", "j", "TensorFlow job dir", flags)
    define_required_str_flag("export_dir", "e", "directory for final Saved Model export", flags)
    define_required_str_flag("eval_export_dir", "eed",
                             "directory to export saved model for model analysis", flags)


def define_standard_tft_flags(flags=None):
    """Register standard tf.transform command line flags"""
    if flags is None:
        flags = tf.flags
    define_required_str_flag("training_data_src", "tds",
                             "paths of the source training data - can be regex", flags)
    define_required_str_flag("training_data_dst", "tdd",
                             "destination for the transformed training data", flags)
    define_required_str_flag("evaluation_data_src", "eds",
                             "paths of the source evaluation data - can be regex", flags)
    define_required_str_flag("evaluation_data_dst", "edd",
                             "destination for the transformed evaluation data", flags)
    define_required_str_flag("transform_fn_dst", "tfd",
                             "destination for the TensorFlow transform function graph", flags)
    define_required_str_flag("temp_location", "tl", "temporary location for temporary files", flags)
    define_required_str_flag("schema_file", "s", "schema file path", flags)
