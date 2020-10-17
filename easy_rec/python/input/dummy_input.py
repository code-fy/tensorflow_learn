#-*- encoding:utf-8 -*-
import tensorflow as tf

if tf.__version__ >= '2.0':
  tf = tf.compat.v1

from easy_rec.python.input.input import Input
from easy_rec.python.protos.dataset_pb2 import DatasetConfig
import logging

class DummyInput(Input):
  ''' dummy input is used to debug the peformance bottleneck of data pipeline '''
  def __init__(self, data_config, feature_config, input_path, task_index=0,
               task_num=1):
    super(DummyInput, self).__init__(data_config, feature_config, input_path,
               task_index, task_num)

  def _build(self, mode, params):
    ''' build fake constant input
    Args:
      mode: tf.estimator.ModeKeys.TRAIN / tf.estimator.ModeKeys.EVAL / tf.estimator.ModeKeys.PREDICT
      params: parameters passed by estimator, currently not used
    Return:
      features tensor dict
      label tensor dict
    '''
    features = {}
    for field, field_type, def_val in zip(self._input_fields,\
                              self._input_field_types,\
                              self._input_field_defaults):
      tf_type = self.get_tf_type(field_type)
      def_val = self.get_type_defaults(field_type, default_val=def_val)
      tensor = tf.constant([def_val] * self._batch_size, dtype=tf_type)
      features[field] = tensor 
    parse_dict = self._preprocess(features)
    return self._get_features(parse_dict), self._get_labels(parse_dict)
