#-*- encoding:utf-8 -*-
import os
import sys
import tensorflow as tf
from easy_rec.python.inference.predictor import Predictor
import logging
from logging import DEBUG
from logging import INFO
import numpy as np
import csv

logging.basicConfig(level=logging.INFO)

class PredictorTest(tf.test.TestCase):  
  def __init__(self, methodName='PredictorTest'):
    super(PredictorTest, self).__init__(methodName=methodName)
    self._predictor = Predictor('data/test/jiguang_export/')
    self._test_path = 'data/test/jiguang_infer.txt'

  def test_pred_list(self):
    with open(self._test_path, 'r') as fin:
      reader = csv.reader(fin)
      inputs = []
      for row in reader:
        inputs.append(row[1:])
      output_res = self._predictor.predict(inputs, batch_size=32)
      assert len(output_res) == 63
      assert abs(output_res[0]['y'] - 0.5726) < 1e-3

  def test_lookup_pred(self):
    lookup_pred_path = 'data/test/lookup_data_test80.csv'
    with open(lookup_pred_path, 'r') as fin:
      reader = csv.reader(fin)
      inputs = []
      for row in reader:
        inputs.append(row[1:])
      predictor = Predictor('data/test/lookup_export')
      output_res = predictor.predict(inputs, batch_size=32)
      assert len(output_res) == 80

  def test_pred_dict(self):
    field_keys = [ "field1", "field2", "field3", "field4", "field5",
                   "field6", "field7", "field8", "field9", "field10",
                   "field11", "field12", "field13", "field14", "field15",
                   "field16", "field17", "field18", "field19", "field20" ]
    with open(self._test_path, 'r') as fin:
      reader = csv.reader(fin)
      inputs = []
      for row in reader:
        inputs.append({ f : row[fid+1] for fid, f in enumerate(field_keys) })
      output_res = self._predictor.predict(inputs, batch_size=32)
      assert len(output_res) == 63
      assert abs(output_res[0]['y'] - 0.5726) < 1e-3

if __name__ == '__main__':
  tf.test.main()
