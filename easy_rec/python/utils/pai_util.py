#!/usr/bin/env python
# -*- encoding:utf-8 -*-
# Filename：util.py
# Author：谦言 (wenmeng.zwm@alibaba-inc.com)
# Date：2019-03-19
# Description：
import sys
import os
import tensorflow as tf
import traceback
import logging

if sys.version_info.major == 2:
  from urllib2 import urlopen, Request, HTTPError
else:
  from urllib.request import urlopen, Request
  from urllib.error import HTTPError

def is_pai_tf():
  try:
    import pai
    logging.info("run on pai tensorflow")
    return True
  except Exception as ex:
    logging.error("import pai exception: %s" % str(ex))
    return False

def download(url):
  _, fname = os.path.split(url)
  request = Request(url=url)
  try:
    response = urlopen(request, timeout=10)
    with open(fname, 'w') as ofile:
      ofile.write(response.read())
    return fname
  except HTTPError as e:
    tf.logging.error('http error: ', e.code)
    tf.logging.error('body:', e.read())
    return None
  except Exception as e:
    tf.logging.error(e)
    tf.logging.error(traceback.format_exc())
    return None


def process_config(configs, task_index=0, worker_num=1):
  """download config and select config for the worker if multiple configs exist
  Args:
    configs: config paths, separated by ','
    task_index: worker index
    worker_num: total number of workers
  """
  configs = configs.split(',')
  if len(configs) > 1:
    assert len(configs) == worker_num, \
      'number of configs must be equal to number of workers, when number of configs > 1'
    config = configs[task_index]
  else:
    config = configs[0]

  if config[:4] == 'http':
    return download(config)
  elif config[:3] == 'oss':
    return config
  else:
    # allow to use this entry file to run experiments from local env
    # to avoid uploading sample file
    return config


def test():
  f = download('http://101374-public.oss-cn-hangzhou-zmf.aliyuncs.com/ssd_vgg16.config')
  assert f == 'ssd_vgg16.config'


if __name__ == "__main__":
  test()
