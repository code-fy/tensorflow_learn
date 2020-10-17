#-*- encoding:utf-8 -*-
import os
import sys
import six
import tensorflow as tf

if tf.__version__ >= '2.0':
  tf = tf.compat.v1

from easy_rec.python.model.ctr_model import CTRModel
from easy_rec.python.feature_column.feature_column import FeatureColumnParser
from easy_rec.python.protos.deepfm_pb2 import DeepFM as DeepFMConfig 
from easy_rec.python.compat import regularizers

class DeepFM(CTRModel):
  def __init__(self, model_config, feature_configs, features,
               labels=None, is_training=False):
    super(DeepFM, self).__init__(model_config, feature_configs, features, 
                                 labels, is_training)
    assert self._model_config.WhichOneof('model') == 'deepfm', \
         'invalid model config: %s' % self._model_config.WhichOneof('model')
    self._model_config = self._model_config.deepfm
    assert isinstance(self._model_config, DeepFMConfig)

    self._wide_features, _ = self._input_layer(self._feature_dict, 'wide')
    self._deep_features, self._fm_features = \
                             self._input_layer(self._feature_dict, 'deep')

  def fm(self):
    fea_dim = self._fm_features[0].get_shape()[1]
    fm_feas = [ tf.expand_dims(x, axis=1) for x in self._fm_features ]
    fm_feas = tf.concat(fm_feas, axis=1)
    sum_square = tf.square(tf.reduce_sum(fm_feas, 1))
    square_sum = tf.reduce_sum(tf.square(fm_feas), 1)
    y_v = 0.5 * tf.subtract(sum_square, square_sum)  

    return y_v

  def dnn(self):
    deep_fea = self._deep_features
    l2reg = regularizers.l2_regularizer(\
                   self._model_config.dense_regularization)
    for i, unit in enumerate(self._model_config.dnn.hidden_units):
      deep_fea = tf.layers.dense(inputs=deep_fea, units=unit,
                          activation=tf.nn.relu,
                          kernel_regularizer=l2reg,
                          name='dnn_%d' % i)
      deep_fea = tf.layers.batch_normalization(deep_fea,
                         training=self._is_training,
                         trainable=True, name='dnn_%d/bn' % i)
    return deep_fea

  def build_predict_graph(self):
    wide_fea_dim = self._wide_features.get_shape()[1]
    wide_fea = tf.layers.dense(self._wide_features,
                          self._model_config.wide_output_dim,
                          kernel_regularizer=regularizers.l1_regularizer(\
                                 self._model_config.wide_regularization),
                          name='wide_feature')
   
    fm_fea = self.fm()

    deep_fea = self.dnn()

    l2reg = regularizers.l2_regularizer(\
                   self._model_config.dense_regularization)
    all_fea = tf.concat([wide_fea, fm_fea,  deep_fea], axis=1)

    for i, unit in enumerate(self._model_config.final_dnn.hidden_units):
      all_fea = tf.layers.dense(all_fea, unit, activation=tf.nn.relu,
                                kernel_regularizer=l2reg,
                                name='all_dnn_%d' % i)
      all_fea = tf.layers.batch_normalization(all_fea,
                               training=self._is_training,
                               trainable=True,
                               name="all_dnn_%d/bn" % i)
    output = tf.layers.dense(all_fea, self._num_class, kernel_regularizer=l2reg,
                             name='output')

    self._add_to_prediction_dict(output)

    return self._prediction_dict
