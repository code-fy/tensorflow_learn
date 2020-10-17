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

class CTRModel(EasyRecModel):
  def __init__(self, model_config, feature_configs, features, labels=None,
               is_training=False):
    super(CTRModel, self).__init__(model_config, feature_configs, features,
                                 labels, is_training)
    self._loss_type = self._model_config.loss_type
    self._num_class = self._model_config.num_class

    if self._labels is not None:
      self._labels = list(self._labels.values())
      if self._loss_type == LossType.CLASSIFICATION:
        if tf.__version__ >= '2.0':
          self._labels[0] = tf.cast(self._labels[0], tf.int64)
        else:
          self._labels[0] = tf.to_int64(self._labels[0])
      elif self._loss_type == LossType.L2_LOSS:
        if tf.__version__ >= '2.0':
          self._labels[0] = tf.cast(self._labels[0], tf.float32)
        else:
          self._labels[0] = tf.to_float(self._labels[0])

  def _add_to_prediction_dict(self, output):
    if self._loss_type == LossType.CLASSIFICATION:
      if self._num_class == 1:
        output = tf.squeeze(output, axis=1)
        probs = tf.sigmoid(output)
        self._prediction_dict['logits'] = output
        self._prediction_dict['probs'] = probs
      else:
        probs = tf.nn.softmax(output, axis=1)
        self._prediction_dict['logits'] = output
        self._prediction_dict['probs'] = probs
        self._prediction_dict['y'] = tf.argmax(output, axis=1)
    else:
      output = tf.squeeze(output, axis=1)
      self._prediction_dict['y'] = output

  def build_loss_graph(self):
    if self._loss_type == LossType.CLASSIFICATION:
      if self._num_class == 1:
        loss = losses.sigmoid_cross_entropy(self._labels[0], \
                              self._prediction_dict['logits'])
        self._loss_dict['cross_entropy_loss'] = loss
      else:
        loss = losses.sparse_softmax_cross_entropy(\
                  labels=self._labels[0], 
                  logits=self._prediction_dict['logits'])
        self._loss_dict['cross_entropy_loss'] = loss
    elif self._loss_type == LossType.L2_LOSS:
      logging.info("l2 loss is used")
      loss = tf.reduce_mean(tf.square(self._labels[0] - self._prediction_dict['y']))
      self._loss_dict['l2_loss'] = loss
    else:
      assert "invalid loss type: %s" % str(self._model_config.loss_type)
    return self._loss_dict

  def build_metric_graph(self, eval_config):
    metric_dict = {}
    for metric in eval_config.metrics_set:
      if metric.WhichOneof('metric') == 'auc':
        assert self._loss_type == LossType.CLASSIFICATION
        assert self._num_class == 1
        metric_dict['auc'] = metrics.auc(self._labels[0],
                                 self._prediction_dict['probs'])
      elif metric.WhichOneof('metric') == 'recall_at_topk':
        assert self._loss_type == LossType.CLASSIFICATION
        assert self._num_class > 1
        metric_dict['recall_at_topk'] = metrics.recall_at_k(\
                      self._labels[0], self._prediction_dict['logits'], 
                      metric.recall_at_topk.topk)
      elif metric.WhichOneof('metric') == 'mean_absolute_error':
        assert self._loss_type == LossType.L2_LOSS
        metric_dict['mean_absolute_error'] = metrics.mean_absolute_error(\
                      self._labels[0], self._prediction_dict['y'])
      elif metric.WhichOneof('metric') == 'accuracy':
        assert self._loss_type == LossType.CLASSIFICATION
        assert self._num_class > 1
        metric_dict['accuracy'] = metrics.accuracy(self._labels[0],
                      self._prediction_dict['y'])
    return metric_dict

  def get_outputs(self):
    if self._loss_type == LossType.CLASSIFICATION:
      if self._num_class == 1:
        return ['probs', 'logits']
      else:
        return ['y', 'probs', 'logits']
    elif self._loss_type == LossType.L2_LOSS:
      return ['y']
