#-*- encoding:utf-8 -*-
import logging
import time
import os
import tensorflow as tf

if tf.__version__ >= '2.0':
  tf = tf.compat.v1

from collections import OrderedDict
from tensorflow.python.saved_model import signature_constants
from easy_rec.python.builders import optimizer_builder
from easy_rec.python.protos.pipeline_pb2 import EasyRecConfig
from easy_rec.python.protos.feature_config_pb2 import FeatureConfig
from easy_rec.python.utils import estimator_utils
from easy_rec.python.protos.train_pb2 import DistributionStrategy
from easy_rec.python.compat import optimizers

class EasyRecEstimator(tf.estimator.Estimator):
  def __init__(self, pipeline_config, model_cls, run_config, params):
    self._pipeline_config = pipeline_config
    self._model_cls = model_cls
    assert isinstance(self._pipeline_config, EasyRecConfig)
      
    super(EasyRecEstimator, self). __init__(
        model_fn=self._model_fn,  model_dir=pipeline_config.model_dir, 
        config=run_config, params=params)
 
  @property
  def feature_configs(self):
    assert len(self._pipeline_config.feature_configs) > 0
    return self._pipeline_config.feature_configs

  @property
  def model_config(self):
    return self._pipeline_config.model_config

  @property
  def eval_config(self):
    return self._pipeline_config.eval_config

  @property
  def train_config(self):
    return self._pipeline_config.train_config

  @property
  def export_config(self):
    return self._pipeline_config.export_config

  def _train_model_fn(self, features, labels, run_config):
    model = self._model_cls(self.model_config, self.feature_configs,
                            features, labels, is_training=True)
    predict_dict = model.build_predict_graph()
    loss_dict = model.build_loss_graph()

    regularization_losses = tf.get_collection(
        tf.GraphKeys.REGULARIZATION_LOSSES)
    if regularization_losses:
      regularization_losses = [reg_loss.get() if hasattr(reg_loss, 'get')
                           else reg_loss for reg_loss in regularization_losses]
      regularization_losses = tf.add_n(
          regularization_losses, name='regularization_loss')
      loss_dict['regularization_loss'] = regularization_losses

      if self.model_config.embedding_regularization > 0:
        all_embeddings = [ x for x in tf.trainable_variables() \
                           if 'embedding_weights:' in x.name ]
        loss_dict['embedding_regularization'] = \
             tf.add_n([ tf.reduce_sum(tf.square(x)) for x in all_embeddings ]) \
             * self.model_config.embedding_regularization
      
    loss = tf.add_n(list(loss_dict.values()))
    loss_dict['total_loss'] = loss
    for key in loss_dict:
      tf.summary.scalar(key, loss_dict[key], family='loss')

    # update op, usually used for batch-norm
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
      update_op = tf.group(*update_ops, name='update_barrier')
      with tf.control_dependencies([update_op]):
        loss = tf.identity(loss, name='total_loss')

    # build optimizer
    optimizer_config = self.train_config.optimizer_config
    optimizer, learning_rate = optimizer_builder.build(optimizer_config)
    tf.summary.scalar('learning_rate', learning_rate[0])

    hooks = []
    # for distributed and synced training
    if self.train_config.sync_replicas and run_config.num_worker_replicas > 1:
      logging.info('sync_replicas: num_worker_replias = %d' % \
                   run_config.num_worker_replicas)
      num_worker_replicas = run_config.num_worker_replicas
      optimizer = tf.train.SyncReplicasOptimizer(
          optimizer,
          replicas_to_aggregate=run_config.num_worker_replicas,
          total_num_replicas=run_config.num_worker_replicas)
      hooks.append(optimizer.make_session_run_hook(run_config.is_chief))
      hooks.append(estimator_utils.ExitBarrierHook(
          run_config.num_worker_replicas, run_config.is_chief,
          self.model_dir))

    gradient_clipping_by_norm = self.train_config.gradient_clipping_by_norm
    if gradient_clipping_by_norm <= 0:
      gradient_clipping_by_norm = None

    # optimize loss
    # colocate_gradients_with_ops=True means to compute gradients
    # on the same device on which op is processes in forward process
    train_op = optimizers.optimize_loss(
        loss=loss,
        global_step=tf.train.get_global_step(),
        learning_rate=None,
        clip_gradients=gradient_clipping_by_norm,
        optimizer=optimizer,
        variables=tf.trainable_variables(),
        summaries=[],
        colocate_gradients_with_ops=True,
        name='')  # Preventing scope prefix on all variables.

    if self.train_config.HasField('fine_tune_checkpoint'):
      fine_tune_ckpt = self.train_config.fine_tune_checkpoint
      logging.warning('will restore from %s' % fine_tune_ckpt)
      fine_tune_ckpt_var_map = self.train_config.fine_tune_ckpt_var_map
      force_restore = self.train_config.force_restore_shape_compatible
      restore_hook = model.restore(fine_tune_ckpt,
          include_global_step=False,
          ckpt_var_map_path=fine_tune_ckpt_var_map,
          force_restore_shape_compatible=force_restore)
      if restore_hook is not None:
        hooks.append(restore_hook)
    # logging
    logging_dict = OrderedDict()
    logging_dict['lr'] = learning_rate[0]
    logging_dict['step'] = tf.train.get_global_step()
    logging_dict.update(loss_dict)
    tensor_order = logging_dict.keys()

    def format_fn(tensor_dict):
      stats = []
      for k in tensor_order:
        tensor_value = tensor_dict[k]
        stats.append('%s = %s' % (k, tensor_value))
      return ','.join(stats)

    log_step_count_steps = self.train_config.log_step_count_steps

    logging_hook = tf.train.LoggingTensorHook(
        logging_dict, every_n_iter=log_step_count_steps,
        formatter=format_fn)
    hooks.append(logging_hook)

    if self.train_config.train_distribute in [\
         DistributionStrategy.CollectiveAllReduceStrategy, \
         DistributionStrategy.MultiWorkerMirroredStrategy ]:
      # for multi worker strategy, we could not replace the
      # inner CheckpointSaverHook, so just use it.
      scaffold = tf.train.Scaffold()
      chief_hooks = []
    else:
      scaffold = tf.train.Scaffold(saver=tf.train.Saver(sharded=False))
      ## saver hook
      saver_hook = estimator_utils.CheckpointSaverHook(
          checkpoint_dir=self.model_dir,
          save_secs=None,
          save_steps=self.train_config.save_checkpoints_steps,
          scaffold=scaffold,
          write_graph=True,
          separate_save=self.train_config.separate_save)
      chief_hooks = [saver_hook]

    # profiling hook
    if self.train_config.is_profiling:
      profile_hook = tf.train.ProfilerHook(\
                       save_steps=log_step_count_steps,
                       output_dir=self.model_dir)
      hooks.append(profile_hook)

    
    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.TRAIN,
        loss=loss,
        predictions=predict_dict,
        train_op=train_op,
        scaffold=scaffold,
        training_chief_hooks=chief_hooks,
        training_hooks=hooks)

  def _eval_model_fn(self, features, labels, run_config):
    start = time.time()
    model = self._model_cls(self.model_config, self.feature_configs,
                            features, labels, is_training=False)
    predict_dict = model.build_predict_graph()
    loss_dict = model.build_loss_graph()
    loss = tf.add_n(list(loss_dict.values()))
    loss_dict['total_loss'] = loss
    metric_dict = model.build_metric_graph(self.eval_config)
    for loss_key in loss_dict.keys():
      loss_tensor = loss_dict[loss_key]
      # add key-prefix to make loss metric key in the same family of train loss
      metric_dict['loss/loss/' + loss_key] = tf.metrics.mean(loss_tensor)
    tf.logging.info("metric_dict keys: %s" % metric_dict.keys())

    end = time.time()
    tf.logging.info('eval graph construct finished. Time %.3fs' % (end - start))
    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.EVAL,
        loss=loss,
        predictions=predict_dict,
        eval_metric_ops=metric_dict)

  def _export_model_fn(self, features, labels, run_config):
    if self.export_config.dump_embedding_shape:
      logging.info("use embedded features as input")
      model = self._model_cls(self.model_config, None, features, labels=None,
                              is_training=False)
    else:
      model = self._model_cls(self.model_config, self.feature_configs, features,
                            labels=None, is_training=False)
    predict_dict = model.build_predict_graph()

    # add output info to estimatorspec
    outputs = {}
    output_list = model.get_outputs()
    for out in output_list:
      assert out in predict_dict, \
        'output node %s not in prediction_dict, can not be exported' % out
      outputs[out] = predict_dict[out]
      tf.logging.info('output %s shape: %s type: %s' %
                      (out, outputs[out].get_shape().as_list(), outputs[out].dtype))
    export_outputs = {
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
          tf.estimator.export.PredictOutput(outputs)
    }
    tf.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS,
                         tf.constant(self._model_dir + "/pipeline.config",
                                     dtype=tf.string, name="pipeline.config"))
    if self.export_config.dump_embedding_shape:
      embed_input_desc_files = tf.gfile.Glob(os.path.join(
                                  self.model_dir, 'embedding_shapes',
                                  "input_layer_*.txt"))
      for one_file in embed_input_desc_files:
        _, one_file_name = os.path.split(one_file)
        tf.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS,  
                 tf.constant(one_file, dtype=tf.string, name=one_file_name))

    return tf.estimator.EstimatorSpec(
             mode=tf.estimator.ModeKeys.PREDICT,
             loss=None, predictions=outputs,
             export_outputs=export_outputs)

  def _model_fn(self, features, labels, mode, 
                              config, params):
    if mode == tf.estimator.ModeKeys.TRAIN:
      return self._train_model_fn(features, labels, config)
    elif mode == tf.estimator.ModeKeys.EVAL:
      return self._eval_model_fn(features, labels, config)
    elif mode == tf.estimator.ModeKeys.PREDICT:
      return self._export_model_fn(features, labels, config) 
tf.device()