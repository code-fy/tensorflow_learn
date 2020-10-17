#-*- encoding:utf-8 -*-
'''
1. Before Run:
sh scripts/gen_proto.sh

2. Train:
2.1 Train 1gpu:
CUDA_VISIBLE_DEVICES=0 python -m easy_rec.python.train_eval --pipeline_config_path samples/model_config/jiguang.config

2.2 Train 2gpu:
* Parameter Server Mode:
  sh scripts/train_ps.sh samples/model_config/jiguang_distribute.config 
* Multi Worker mode:
  sh scripts/train_multi_worker.sh samples/model_config/jiguang_multi_worker.config

2.3 Train on pai:
  cd pai_jobs
  sh deploy.sh
  odpscmd -f train.sql

2.4 Train on pai outside:
  cd pai_jobs
  sh deploy_ext.sh
  odpscmd -f train_ext.sql

3 Eval
3.1 Eval Local:
  CUDA_VISIBLE_DEVICES=0 python -m easy_rec.python.eval --pipeline_config_path samples/model_config/jiguang.config

3.2 Eval On pai:
  odpscmd -f eval.sql

4 Export
4.1 Export local:
  CUDA_VISIBLE_DEVICES='' python -m easy_rec.python.export --pipeline_config_path samples/model_config/jiguang.config --export_dir tmp_export

5 Test
5.1 unit test:
  CUDA_VISIBLE_DEVICES=0 python -m easy_rec.python.test.run 
5.2 end2end test:
  sh scripts/end2end_test.sh
5.3 test data path: data/test/
  if you add new data, please do the following to commit it to git-lfs before "git commit":
    python git-lfs/git_lfs.py add data/test/new_data
    python git-lfs/git_lfs.py push

6 Create config from excel:
  python scripts/create_config_from_excel.py data/test/dwd_douyu.xls dwd_douyu.config

7 Export 
  CUDA_VISIBLE_DEVICES=6 python -m easy_rec.python.export --pipeline_config_path samples/model_config/jiguang.config --export_dir jiguang_export_v2

8 Inference:
  ```
    import csv
    from easy_rec.python.inference.predictor import Predictor
    predictor = Predictor(SAVED_MODEL_DIR)
    with open(INPUT_CSV, 'r') as fin:
      reader = csv.reader(fin)
      inputs = []
      for row in reader:
        inputs.append(row[1:])
      output_res = self._predictor.predict(inputs, batch_size=32)
  ```
  ```
    import csv
    from easy_rec.python.inference.predictor import Predictor
    predictor = Predictor(SAVED_MODEL_DIR)
    field_keys = [ "field1", "field2", "field3", "field4", "field5",
                   "field6", "field7", "field8", "field9", "field10",
                   "field11", "field12", "field13", "field14", "field15",
                   "field16", "field17", "field18", "field19", "field20" ]
    with open(INPUT_CSV, 'r') as fin:
      reader = csv.reader(fin)
      inputs = []
      for row in reader:
        inputs.append({ f : row[fid+1] for fid, f in enumerate(field_keys) })
      output_res = self._predictor.predict(inputs, batch_size=32)
  ```

9 Build Pip package:
  python setup.py sdist bdist_wheel

10 Export large embedding into partitions for eas serving:
  # make sure that export_config.dump_embedding_shape is set to true
  CUDA_VISIBLE_DEVICES='' python -m easy_rec.python.export --pipeline_config_path samples/model_config/douyu_tower.config --export_dir douyu_export
  PYTHONPATH=. python scripts/create_eas_serving_config.py douyu_export/1597127233 douyu_eas.config

11 Hyperparameter tuning demo
On Pai:
  python -m easy_rec.python.hpo.pai_hpo --odps_config pai_jobs/odps_config.ev --oss_config ~/.ossutilconfig_pai_vision_data_sh --bucket oss://pai-vision-data-sh --role_arn acs:ram::1217060697188167:role/ev-ext-test-oss --hyperparams samples/hpo/hyperparams.json  --exp_dir easy_rec_test/experiment/model_v0  --tables train_longonehot_4deepfm_20,test_longonehot_4deepfm_20 --algo_proj_name easy_vision_test --config_path oss://pai-vision-data-sh/easy_rec_test/jiguang_odps_hpo_v2.config
On EMR:
  python -m easy_rec.python.hpo.emr_hpo --hyperparams hyperparams.json  --config_path ./dwd_avazu_ctr_deepmodel.config --exp_dir hdfs:///user/easy_rec_test/experiment/hpo_test_v0
'''

import sys
import os

curr_dir, _ = os.path.split(__file__)
parent_dir = os.path.dirname(curr_dir)
sys.path.insert(0, parent_dir)

__version__ = '20201012'

print("easy_rec version: %s" % __version__)
print("Usage: easy_rec.help()")

import logging
logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')

from easy_rec.python.main import train_and_evaluate
from easy_rec.python.main import evaluate
from easy_rec.python.main import export

from easy_rec.python.inference.predictor import Predictor

_global_config = {}

sample_file_path='https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/easy-rec/samples/easy_rec_samples_20201012.tar.gz'

def download_samples():
  os.mkdir('easy_rec_sample')
  os.chdir('easy_rec_sample')
  os.system('wget %s -O easy_rec_samples.tar.gz' % sample_file_path)
  os.system('tar -zvxf easy_rec_samples.tar.gz')
  os.system('rm -rf easy_rec_samples.tar.gz')
  print("Usage: easy_rec.help()")

def help():
  print('1. Train:')
  print('1.1 Train 1gpu:')
  print('\tCUDA_VISIBLE_DEVICES=0 python -m easy_rec.python.train_eval --pipeline_config_path samples/model_config/jiguang.config')
  print('1.2 Train 2gpu:')
  print('\tsh scripts/train_2gpu.sh samples/model_config/jiguang_distribute.config')
  
  print('2. Eval:')
  print('\tCUDA_VISIBLE_DEVICES=0 python -m easy_rec.python.eval --pipeline_config_path samples/model_config/jiguang.config')

  print('3 Export')
  print('\tCUDA_VISIBLE_DEVICES='' python -m easy_rec.python.export --pipeline_config_path samples/model_config/jiguang.config --export_dir tmp_export')

  print('4 Create config from excel:')
  print('\tpython scripts/create_config_from_excel.py data/test/dwd_douyu.xls dwd_douyu.config')

  print('5. Inference:')
  print(\
'''
    # use list input
    import csv
    from easy_rec.python.inference.predictor import Predictor
    predictor = Predictor(SAVED_MODEL_DIR)
    with open(INPUT_CSV, 'r') as fin:
      reader = csv.reader(fin)
      inputs = []
      for row in reader:
        inputs.append(row[1:])
      output_res = self._predictor.predict(inputs, batch_size=32)

    # use dict input
    import csv
    from easy_rec.python.inference.predictor import Predictor
    predictor = Predictor(SAVED_MODEL_DIR)
    field_keys = [ "field1", "field2", "field3", "field4", "field5",
                   "field6", "field7", "field8", "field9", "field10",
                   "field11", "field12", "field13", "field14", "field15",
                   "field16", "field17", "field18", "field19", "field20" ]
    with open(INPUT_CSV, 'r') as fin:
      reader = csv.reader(fin)
      inputs = []
      for row in reader:
        inputs.append({ f : row[fid+1] for fid, f in enumerate(field_keys) })
      output_res = self._predictor.predict(inputs, batch_size=32)
''')

  if os.path.exists('easy_rec_sample'):
    if os.path.exists('easy_rec_sample/samples') and \
       os.path.exists('easy_rec_sample/data'):
      return
  if os.getcwd().endswith('easy_rec_sample'):
    if os.path.exists(os.path.join(os.getcwd(), 'samples')) and \
       os.path.exists(os.path.join(os.getcwd(), 'data')): 
      return
  print("Download samples: easy_rec.download_samples()")
