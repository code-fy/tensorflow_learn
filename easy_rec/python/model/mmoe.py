#-*- encoding:utf-8 -*-
import os
import sys
import six
import tensorflow as tf

if tf.__version__ >= '2.0':
  tf = tf.compat.v1
  losses = tf.compat.v1.losses
  metrics = tf.compat.v1.metrics
else:
  losses = tf.losses
  metrics = tf.metrics

from easy_rec.python.model.multi_task_model import MultiTaskModel
from easy_rec.python.feature_column.feature_column import FeatureColumnParser
from easy_rec.python.protos.mmoe_pb2 import MMoE as MMoEConfig
from easy_rec.python.compat import regularizers
from easy_rec.python.protos.loss_pb2 import LossType
import logging

class MMoE(MultiTaskModel):
  def __init__(self, model_config, feature_configs, features,
               labels=None, is_training=False):
    super(MMoE, self).__init__(model_config, feature_configs, features,
                                 labels, is_training)
    assert self._model_config.WhichOneof('model') == 'mmoe', \
         'invalid model config: %s' % self._model_config.WhichOneof('model')
    self._model_config = self._model_config.mmoe
    assert isinstance(self._model_config, MMoEConfig)

    self._features, _ = self._input_layer(self._feature_dict, 'all')
    self._expert_num = len(self._model_config.experts)
    self._task_num = len(self._model_config.task_towers)
    self._towers_name = [i.input for i in self._model_config.task_towers]

    logging.info('expert_num: {0}'.format(self._expert_num))
    logging.info('task_num: {0}'.format(self._task_num))
    logging.info('tower_name: {0}'.format(self._towers_name))

    self._weight_list = []
    for i in range(self._task_num):
      task_config = self._model_config.task_towers[i]
      self._num_class_list.append(task_config.num_class)
      self._loss_type_list.append(task_config.loss_type)
      self._metrics_list.append(task_config.metrics_set)
      self._weight_list.append(task_config.weight)

    if self._labels is not None:
      self._labels = list(self._labels.values())
      for i in range(self._task_num):
        if self._loss_type_list[i] == LossType.CLASSIFICATION:
          if tf.__version__ >= '2.0':
            self._labels[i] = tf.cast(self._labels[i], tf.int64)
          else:
            self._labels[i] = tf.to_int64(self._labels[i])
        elif self._loss_type_list[i] == LossType.L2_LOSS:
          if tf.__version__ >= '2.0':
            self._labels[i] = tf.cast(self._labels[i], tf.float32)
          else:
            self._labels[i] = tf.to_float(self._labels[i])

      assert self._task_num == len(self._labels), \
          'task num (%d) is inconsistent with target num (%d)' % \
          (self._task_num, len(self._labels))

    self._l2_reg = regularizers.l2_regularizer(
                        self._model_config.l2_regularization)

  def dnn(self, dnn_config, deep_fea, name):
    for i, unit in enumerate(dnn_config.hidden_units):
      deep_fea = tf.layers.dense(inputs=deep_fea, units=unit,
                                 kernel_regularizer=self._l2_reg,
                                 name='%s/dnn_%d' % (name, i))
      deep_fea = tf.layers.batch_normalization(deep_fea,
                                               training=self._is_training,
                                               trainable=True, name='%s/dnn_%d/bn' % (name, i))
      deep_fea = tf.nn.relu(deep_fea)
    return deep_fea

  def gate(self, unit, deep_fea, name):
    fea = tf.layers.dense(inputs=deep_fea, units=unit,
                          kernel_regularizer=self._l2_reg,
                          name='%s/dnn' % name)
    fea = tf.nn.softmax(fea, axis=1)
    return fea

  def build_loss_graph(self):
    for i in range(len(self._labels)):
      task_name = self._towers_name[i]
      task_weight = self._weight_list[i]
      if self._loss_type_list[i] == LossType.CLASSIFICATION:
        if self._num_class_list[i] == 1:
          loss = losses.sigmoid_cross_entropy(self._labels[i], \
                                self._prediction_dict['logits_%s' % task_name])
        else:
          loss = losses.sparse_softmax_cross_entropy(\
                    labels=self._labels[i],
                    logits=self._prediction_dict['logits_%s' % task_name])
        self._loss_dict['weighted_cross_entropy_loss_%s' % task_name] = task_weight * loss
      elif self._loss_type_list[i] == LossType.L2_LOSS:
        logging.info("l2 loss is used")
        loss = tf.reduce_mean(tf.square(self._labels[i] - self._prediction_dict['y_%s' % task_name]))
        self._loss_dict['l2_loss_%s' % task_name] = loss
      else:
        assert "invalid loss type: %s" % str(self._model_config.loss_type)
    return self._loss_dict

  def build_predict_graph(self):
    tower_outputs = []
    expert_fea_arr = []
    for expert_id in range(self._expert_num):
      expert_config = self._model_config.experts[expert_id]
      expert_fea = self.dnn(expert_config.dnn, self._features, name=expert_config.input)
      expert_fea_arr.append(expert_fea)
    experts_fea = tf.stack(expert_fea_arr, axis=1)

    for task_tower_id in range(self._task_num):
      tower_config = self._model_config.task_towers[task_tower_id]
      gate = self.gate(self._expert_num, self._features, name=self._towers_name[task_tower_id] + '_gate')
      gate = tf.expand_dims(gate, -1)
      experts_output = tf.multiply(experts_fea, gate)
      experts_output = tf.reshape(experts_output, (-1, experts_output.shape[1] * experts_output.shape[2]))

      tower_output = self.dnn(tower_config.dnn, experts_output, name=self._towers_name[task_tower_id])
      tower_output = tf.layers.dense(inputs=tower_output, units=1, kernel_regularizer=self._l2_reg,
                                     name='dnn_output_%d' % task_tower_id)

      tower_outputs.append(tower_output)
    self._add_to_prediction_dict(tower_outputs)
    return self._prediction_dict
