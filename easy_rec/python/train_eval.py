import tensorflow as tf

if tf.__version__ >= '2.0':
  tf = tf.compat.v1

import logging
from logging import DEBUG
from logging import INFO
from easy_rec.python.main import train_and_evaluate, _train_and_evaluate_impl
from easy_rec.python.utils import config_util, hpo_util

logging.basicConfig(
  format='[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d : %(message)s',
  level=logging.INFO)
tf.app.flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config '
                                                         'file.')
tf.app.flags.DEFINE_bool('continue_train', False, 'continue train using existing '
                                                  'model dir')
tf.app.flags.DEFINE_integer('task_index', 0, 'task index passed by pai')
tf.app.flags.DEFINE_string('worker_hosts', '', 'worker hosts passed by pai')
tf.app.flags.DEFINE_string('job_name', '', help='job_name passed by pai')
tf.app.flags.DEFINE_string('ps_hosts', '', help='ps_hosts passed by pai')
tf.app.flags.DEFINE_string('hpo_param_path', None, help='hyperparam tuning param path')
tf.app.flags.DEFINE_string("hpo_metric_save_path", None, help="hyperparameter save metric path")
tf.app.flags.DEFINE_string("model_dir", None, help="will update the model_dir in pipeline_config")

FLAGS = tf.app.flags.FLAGS

def main(argv):
  try:
    import pai
    from easy_rec.python.utils.pai_tf_config import pai_tf_config
    pai_tf_config(FLAGS.task_index, FLAGS.job_name, FLAGS.ps_hosts,
                  FLAGS.worker_hosts) 
  except Exception as ex:
    logging.info('import pai exception: %s' % str(ex))
    pass
  if FLAGS.pipeline_config_path is not None:
    pipeline_config = config_util.get_configs_from_pipeline_file(\
                        FLAGS.pipeline_config_path, False)
    if FLAGS.model_dir:
      pipeline_config.model_dir = FLAGS.model_dir
      print('update model_dir to %s' % pipeline_config.model_dir)
    if FLAGS.hpo_param_path is not None:
      config_util.edit_config(pipeline_config, FLAGS.hpo_param_path)
      config_util.auto_expand_share_feature_configs(pipeline_config) 
      _train_and_evaluate_impl(pipeline_config, FLAGS.continue_train)
      hpo_util.save_eval_metrics(pipeline_config.model_dir,
                                 metric_save_path=FLAGS.hpo_metric_save_path,
                                 has_evaluator=False)
    else:
      config_util.auto_expand_share_feature_configs(pipeline_config) 
      _train_and_evaluate_impl(pipeline_config, FLAGS.continue_train)
  else:
    raise ValueError("pipeline_config_path should not be empty when training!")

if __name__ == '__main__':
  tf.app.run()
