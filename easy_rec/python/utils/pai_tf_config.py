#-*- encoding:utf-8 -*-
import os
import sys
import tensorflow as tf
import json
import logging

def pai_tf_config(task_index, job_name, ps_hosts, worker_hosts):
  logging.info('task_index = %d job_name = %s ps_hosts = %s worker_hosts = %s' % (task_index, job_name, ps_hosts, worker_hosts))
  if task_index > 1:
    task_index = task_index - 1
        
  ps_hosts = ps_hosts.split(",")
  worker_hosts = worker_hosts.split(",")
  ps_num = len(ps_hosts)
  worker_num = len(worker_hosts) - 1

  logging.info('pai_tf_config:ps_num = %d worker_num = %d' % (ps_num, worker_num))
  logging.info('pai_tf_config:original TF_CONFIG = %s' % os.environ['TF_CONFIG'])
  
  ##if len(worker_hosts):
  ##  cluster = {"chief": [worker_hosts[0]], "ps": ps_hosts,
  ##             "worker": worker_hosts[2:]}
  ##  if job_name == "ps":
  ##    os.environ['TF_CONFIG'] = json.dumps(
  ##            {'cluster': cluster,
  ##             'task': {'type': job_name,
  ##                      'index': task_index}
  ##            })
  ##  elif job_name == "worker":
  ##    if task_index == 0:
  ##      os.environ['TF_CONFIG'] = json.dumps(
  ##              {'cluster': cluster, 'task': {'type': "chief", 'index': 0}})
  ##    elif task_index == 1:
  ##      os.environ['TF_CONFIG'] = json.dumps(
  ##              {'cluster': cluster, 'task': {'type': "evaluator", 'index': 0}})
  ##    else:
  ##      os.environ['TF_CONFIG'] = json.dumps(
  ##              {'cluster': cluster, 
  ##               'task': {'type': job_name,
  ##                        'index': task_index - 2
  ##                       }
  ##              })
