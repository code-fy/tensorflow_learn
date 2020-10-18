#-*- encoding:utf-8 -*-
import os
import sys
import six
import tensorflow as tf
from easy_rec.python.compat import regularizers
from easy_rec.python.model.easy_rec_model import EasyRecModel
from easy_rec.python.feature_column.feature_column import FeatureColumnParser
from easy_rec.python.protos.dssm_pb2 import DSSM as DSSMConfig

if tf.__version__ >= '2.0':
  tf = tf.compat.v1

class DSSM(EasyRecModel):
  def __init__(self, model_config, feature_configs, features,
               labels=None, is_training=False):
    super(DSSM, self).__init__(model_config, feature_configs, features,
                                 labels, is_training)
    assert self._model_config.WhichOneof('model') == 'dssm', \
         'invalid model config: %s' % self._model_config.WhichOneof('model')
    self._model_config = self._model_config.dssm
    assert isinstance(self._model_config, DSSMConfig)

    if self._labels is not None:
      self._labels = list(self._labels.values())

    self.user_tower = self._model_config.user_tower
    self.user_tower_feature, _ = self._input_layer(self._feature_dict, 'user')
    self.user_id = self.user_tower.id
    self.item_tower = self._model_config.item_tower
    self.item_tower_feature, _ = self._input_layer(self._feature_dict, 'item')
    self.item_id = self.item_tower.id

    self._l2_reg = regularizers.l2_regularizer(\
                         self._model_config.l2_regularization)

  def dnn(self, dnn_config, deep_fea, name):
    with tf.variable_scope(name_or_scope=name, reuse=tf.AUTO_REUSE):
      for i, unit in enumerate(dnn_config.hidden_units):
        deep_fea = tf.layers.dense(inputs=deep_fea, units=unit,
                                   kernel_regularizer=self._l2_reg,
                                   name='dnn_%d' % i)
        # deep_fea = tf.layers.dense(inputs=deep_fea, units=unit,
        #                            kernel_initializer=tf.truncated_normal_initializer(0,0.1),
        #                            kernel_regularizer=self._l2_reg,
        #                            name='dnn_%d' % i)

        deep_fea = tf.layers.batch_normalization(deep_fea,
                                                 training=self._is_training,
                                                 trainable=True, name='dnn_%d/bn' % i)
        deep_fea = tf.nn.relu(deep_fea)

      fea_ = tf.norm(deep_fea, axis=1, keepdims=True)
      fea_norm = tf.div(deep_fea, fea_)
    return fea_norm

  def sim(self, user_emb, item_emb):
      user_item_sim = tf.reduce_sum(tf.multiply(user_emb, item_emb), axis=1)
      return user_item_sim

  def build_predict_graph(self):
    user_tower_emb = self.dnn(self.user_tower.dnn,  self.user_tower_feature, name='user_dnn')
    item_tower_emb = self.dnn(self.item_tower.dnn, self.item_tower_feature, name='item_dnn')
    y_pred = self.sim(user_tower_emb, item_tower_emb)
    self._prediction_dict['y'] = y_pred
    self._prediction_dict['user_emb'] = tf.reduce_join(tf.as_string(user_tower_emb), axis = -1, separator = ",")
    self._prediction_dict['item_emb'] = tf.reduce_join(tf.as_string(item_tower_emb), axis = -1, separator = ",")
    return self._prediction_dict

  def build_loss_graph(self):
    loss = tf.losses.log_loss(self._labels[0], self._prediction_dict['y'])
    self._loss_dict['cross_entropy_loss'] = loss
    return self._loss_dict

  def build_metric_graph(self, eval_config):
    metric_dict = {}
    for metric in eval_config.metrics_set:
      if metric.WhichOneof('metric') == 'auc':
        metric_dict['auc'] = tf.metrics.auc(self._labels[0],
                                 self._prediction_dict['y'])
      elif metric.WhichOneof('metric') == 'recall_at_topk':
        metric_dict['recall_at_topk'] = tf.metrics.recall_at_topk(\
                      self._labels[0], self._prediction_dict['y'],
                      metric.recall_at_topk.topk)
    return metric_dict

  def get_outputs(self):
    return ['user_emb', 'item_emb']
