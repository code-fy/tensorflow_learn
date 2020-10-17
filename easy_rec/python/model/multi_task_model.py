#-*- encoding:utf-8 -*-
import os
import sys
import six
import logging
import tensorflow as tf
from easy_rec.python.model.easy_rec_model import EasyRecModel
from easy_rec.python.protos.loss_pb2 import LossType

if tf.__version__ >= "2.0":
  losses = tf.compat.v1.losses
  metrics = tf.compat.v1.metrics
else:
  losses = tf.losses
  metrics = tf.metrics

class MultiTaskModel(EasyRecModel):
  def __init__(self, model_config, feature_configs, features, labels=None,
               is_training=False):
    super(MultiTaskModel, self).__init__(model_config, feature_configs, features,
                                 labels, is_training)
    self._loss_type_list = []
    self._num_class_list = []
    self._metrics_list = []

  def _add_to_prediction_dict(self, output):
    for task_tower_id in range(self._task_num):
      task_name = self._towers_name[task_tower_id]
      if self._loss_type_list[task_tower_id] == LossType.CLASSIFICATION:
        if self._num_class_list[task_tower_id] == 1:
          output_ = tf.squeeze(output[task_tower_id], axis=1)
          probs_ = tf.sigmoid(output_)
          self._prediction_dict['logits_%s' % task_name] = output_
          self._prediction_dict['probs_%s' % task_name] = probs_
        else:
          output_ = output[task_tower_id]
          probs_ = tf.nn.softmax(output_, axis=1)
          self._prediction_dict['logits_%s' % task_name] = output_
          self._prediction_dict['probs_%s' % task_name] = probs_
          self._prediction_dict['y_%s' % task_name] = tf.argmax(output_, axis=1)
      else:
        output_ = tf.squeeze(output[task_tower_id], axis=1)
        self._prediction_dict['y_%s' % task_name] = output_

  def build_metric_graph(self, eval_config):
    metric_dict = {}
    for task_tower_id in range(self._task_num):
      task_name = self._towers_name[task_tower_id]
      task_metrics = self._metrics_list[task_tower_id]
      for metric in task_metrics:
        if metric.WhichOneof('metric') == 'auc':
          assert self._loss_type_list[task_tower_id] == LossType.CLASSIFICATION
          assert self._num_class_list[task_tower_id] == 1
          metric_dict['auc_%s' % task_name] = metrics.auc(self._labels[task_tower_id],
                                   self._prediction_dict['probs_%s' % task_name])
        elif metric.WhichOneof('metric') == 'recall_at_topk':
          assert self._loss_type_list[task_tower_id] == LossType.CLASSIFICATION
          assert self._num_class_list[task_tower_id] > 1
          metric_dict['recall_at_topk_%s' % task_name] = metrics.recall_at_k(\
                        self._labels[task_tower_id], self._prediction_dict['logits_%s' % task_name],
                        metric.recall_at_topk.topk)
        elif metric.WhichOneof('metric') == 'mean_absolute_error':
          assert self._loss_type_list[task_tower_id] == LossType.L2_LOSS
          metric_dict['mean_absolute_error_%s' % task_name] = metrics.mean_absolute_error(\
                        self._labels[task_tower_id], self._prediction_dict['y_%s' % task_name])
        elif metric.WhichOneof('metric') == 'accuracy':
          assert self._loss_type_list[task_tower_id] == LossType.CLASSIFICATION
          assert self._num_class_list[task_tower_id] > 1
          metric_dict['accuracy_%s' % task_name] = metrics.accuracy(self._labels[task_tower_id],
                        self._prediction_dict['y_%s' % task_name])
    return metric_dict

  def get_outputs(self):
    outputs = []
    for task_tower_id in range(self._task_num):
      task_name = self._towers_name[task_tower_id]
      if self._loss_type_list[task_tower_id] == LossType.CLASSIFICATION:
        if self._num_class_list[task_tower_id] == 1:
          outputs.extend(['probs_%s' % task_name, 'logits_%s' % task_name])
        else:
          outputs.extend(['y_%s' % task_name, 'probs_%s' % task_name, 'logits_%s' % task_name])
      elif self._loss_type_list[task_tower_id] == LossType.L2_LOSS:
        outputs.extend('y_%s' % task_name)
    return outputs