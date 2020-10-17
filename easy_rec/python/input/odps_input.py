#-*- encoding:utf-8 -*-
import tensorflow as tf
from easy_rec.python.input.input import Input

class OdpsInput(Input):
  def __init__(self, data_config, feature_config, input_path, task_index=0,
               task_num=1):
    super(OdpsInput, self).__init__(data_config, feature_config, input_path,
                                    task_index, task_num)

  def _build(self, mode, params):
    print('task_num = %d task_index = %d' % (self._task_num, self._task_index))
    reader = tf.TableRecordReader(csv_delimiter=self._data_config.separator,
                                  slice_count=self._task_num, 
                                  slice_id=self._task_index)
    if type(self._input_path) != list:
      self._input_path = [ self._input_path ]
    if mode == tf.estimator.ModeKeys.TRAIN:
      file_queue = tf.train.string_input_producer(self._input_path,
                     num_epochs=self.num_epochs, capacity=1000,
                     shuffle=True)
    else:
      file_queue = tf.train.string_input_producer(self._input_path,
                     num_epochs=1, capacity=1000,
                     shuffle=False)
    key, value = reader.read_up_to(file_queue, self._batch_size)

    record_defaults = self._input_field_defaults
    fields = tf.decode_csv(value, 
                          record_defaults=record_defaults, 
                          field_delim=self._data_config.separator, 
                          name='decode_csv')
  
    inputs = { self._input_fields[x] : fields[x] for x in self._effective_fids }
    for x in self._label_fids:
      inputs[self._input_fields[x]] = fields[x] 

    fields = self._preprocess(inputs)

    features = self._get_features(fields)
    #import pai
    if mode != tf.estimator.ModeKeys.PREDICT:
      labels = self._get_labels(fields)
      #features, labels = pai.data.prefetch(features=(features, labels),
      #                         capacity=self._prefetch_size, num_threads=2,
      #         closed_exception_types=(tuple([tf.errors.InternalError])))
      return features, labels
    else:
      #features = pai.data.prefetch(features=(features,),
      #                         capacity=self._prefetch_size, num_threads=2,
      #         closed_exception_types=(tuple([tf.errors.InternalError])))
      return features
  
