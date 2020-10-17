# -*- coding:utf-8 -*-
# Author: mengli.cml@alibaba-inc.com
# Date: 2018-06-22
# define cv_input, the base class for cv tasks

from easy_rec.python.utils import hpo_util
import tensorflow as tf
import numpy as np
import json
import os
import logging
import time
import shutil
from logging import DEBUG
from logging import INFO
from easy_rec.python.utils import config_util

logging.basicConfig(level=logging.INFO)

class HPOTest(tf.test.TestCase):  
  def __init__(self, methodName='HPOTest'):
    super(HPOTest, self).__init__(methodName = methodName)
    self._metric_data_path = "data/test/hpo_test/eval_val/*.tfevents.*"

  def test_get_metric(self):
    vals = hpo_util.get_all_eval_result(self._metric_data_path)
    logging.info('eval result num = %d' % len(vals))
    logging.info('eval result[0] = %s' % json.dumps(vals[0]))

  def test_save_eval_metrics(self):
    tmp_file = '/tmp/easy_rec_hpo_test_%d.metric' % time.time()
    hpo_util.save_eval_metrics('data/test/hpo_test/', tmp_file, False)
    os.remove(tmp_file)

  def test_edit_config(self):
    tmp_file = 'samples/model_config/dwd_avazu_ctr_multi_cls.config'
    tmp_config = config_util.get_configs_from_pipeline_file(tmp_file)
    tmp_file = 'samples/hpo/hpo_param.json'
    tmp_config = config_util.edit_config(tmp_config, tmp_file)  
    assert tmp_config.feature_configs[0].embedding_dim == 120

  def test_edit_config_v2(self):
    tmp_file = 'samples/model_config/dwd_avazu_ctr_multi_cls.config'
    tmp_config = config_util.get_configs_from_pipeline_file(tmp_file)
    tmp_file = 'samples/hpo/hpo_param_v2.json'
    tmp_config = config_util.edit_config(tmp_config, tmp_file)  
    for tmp_fea in tmp_config.feature_configs:
      if tmp_fea.input_names[0] == 'site_id':
        assert tmp_fea.embedding_dim == 32
      else:
        assert tmp_fea.embedding_dim == 16

  def test_edit_config_v3(self):
    tmp_file = 'samples/model_config/dwd_avazu_ctr_multi_cls.config'
    tmp_config = config_util.get_configs_from_pipeline_file(tmp_file)
    tmp_file = 'samples/hpo/hpo_param_v3.json'
    tmp_config = config_util.edit_config(tmp_config, tmp_file)  
    for i, tmp_fea in enumerate(tmp_config.feature_configs):
      if i >= 10 and i < 20:
        assert tmp_fea.embedding_dim == 37
      else:
        assert tmp_fea.embedding_dim == 16

  def test_edit_config_v4(self):
    tmp_file = 'samples/model_config/dwd_avazu_ctr_multi_cls.config'
    tmp_config = config_util.get_configs_from_pipeline_file(tmp_file)
    tmp_file = 'samples/hpo/hpo_param_v4.json'
    tmp_config = config_util.edit_config(tmp_config, tmp_file)  
    for i, tmp_fea in enumerate(tmp_config.feature_configs):
      if i < 15:
        assert tmp_fea.embedding_dim == 37
      else:
        assert tmp_fea.embedding_dim == 16

  def test_edit_config_v5(self):
    tmp_file = 'samples/model_config/dwd_avazu_ctr_multi_cls.config'
    tmp_config = config_util.get_configs_from_pipeline_file(tmp_file)
    tmp_file = 'samples/hpo/hpo_param_v5.json'
    tmp_config = config_util.edit_config(tmp_config, tmp_file)  
    for i, tmp_fea in enumerate(tmp_config.feature_configs):
      if i >= 5:
        assert tmp_fea.embedding_dim == 37
      else:
        assert tmp_fea.embedding_dim == 16

  def test_edit_config_v6(self):
    tmp_file = 'samples/model_config/dwd_avazu_ctr_multi_cls.config'
    tmp_config = config_util.get_configs_from_pipeline_file(tmp_file)
    tmp_file = 'samples/hpo/hpo_param_v6.json'
    tmp_config = config_util.edit_config(tmp_config, tmp_file)  
    for i, tmp_fea in enumerate(tmp_config.feature_configs):
      if tmp_fea.input_names[0] >= 'site':
        assert tmp_fea.embedding_dim == 32, 'input_name = %s %d' % (tmp_fea.input_names[0], tmp_fea.embedding_dim)
      else:
        assert tmp_fea.embedding_dim == 16

  def test_save_eval_metrics(self):
    os.environ['TF_CONFIG'] = '''
{ "cluster": { 
    "worker": ["127.0.0.1:2020"],
    "chief": ["127.0.0.1:2021"] 
   },
   "task": {"type": "chief", "index": 0}
}
'''
    tmp_file = '/tmp/easy_rec_hpo_test_%d.metric' % time.time()
    hpo_util.save_eval_metrics('data/test/hpo_test/', tmp_file, False)
    os.remove(tmp_file)
 

if __name__ == '__main__':
  tf.test.main()
