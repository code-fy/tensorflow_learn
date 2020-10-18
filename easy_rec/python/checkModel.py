#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import constants
from tensorflow.core.protobuf import meta_graph_pb2
import logging
import time
import numpy as np

graph = tf.Graph()
with graph.as_default():
    session_config = tf.ConfigProto()
    session_config.intra_op_parallelism_threads=12
    session_config.inter_op_parallelism_threads=12
    
    with tf.Session(config=session_config) as sess:
        meta_graph_def = tf.saved_model.loader.load(
          sess, [tf.saved_model.tag_constants.SERVING], 
          # '/Users/gaoyue/Documents/展心展力/deepfm上线/model/1602739558/'
          #'/Users/dawn/Documents/极光/shuangta/savemodel_small/'
            '/Users/hycao/tensorflow_learn/path/to/model'
          )
        
        signature_def = meta_graph_def.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        inputs = signature_def.inputs
        feed_dict = {}

        print(len(inputs.items()))
        for name, tensor in inputs.items():
            print('Load input binding: %s -> %s' % (name, tensor.name))
            print(graph.get_tensor_by_name(tensor.name))
        outputs = signature_def.outputs
        fetch_dict = {}
        for name, tensor in outputs.items():
            print('Load output binding: %s -> %s' % (name, tensor.name))
