#-*- encoding:utf-8 -*-
import tensorflow as tf
import logging
from easy_rec.python.protos.feature_config_pb2 import FeatureConfig, WideOrDeep
from easy_rec.python.compat.feature_column import feature_column_v2 as feature_column
from easy_rec.python.compat.feature_column import sequence_feature_column

if tf.__version__ >= "2.0":
  min_max_variable_partitioner = tf.compat.v1.min_max_variable_partitioner
else:
  min_max_variable_partitioner = tf.min_max_variable_partitioner

class FeatureKeyError(KeyError):
  def __init__(self, feature_name):
    super(FeatureKeyError, self).__init__(feature_name)
    
class FeatureColumnParser(object):
  def __init__(self, feature_configs, wide_deep_dict={}):
    self._feature_configs = feature_configs
    self._wide_deep_dict = wide_deep_dict
    self._deep_columns = {}
    self._wide_columns = {}

    self._share_embed_names = {}
    self._share_embed_dims = {}
    self._share_embed_partitions = {}
    for config in self._feature_configs:
      if not config.HasField('embedding_name'):
        continue
      embed_dim = config.embedding_dim
      embed_name = config.embedding_name
      if embed_name in self._share_embed_names:
        self._share_embed_names[embed_name] += 1
        if embed_dim > self._share_embed_dims[embed_name]:
          self._share_embed_dims[embed_name] = embed_dim
      else: 
        self._share_embed_names[embed_name] = 1
        self._share_embed_dims[embed_name] = embed_dim
      if embed_name not in self._share_embed_partitions or \
         self._share_embed_partitions[embed_name] is None: 
        self._share_embed_partitions[embed_name] =\
             self._build_partitioner(config)

    # remove not shared embedding names
    not_shared = [ x for x in self._share_embed_names \
                     if self._share_embed_names[x] == 1 ]
    for embed_name in not_shared:
      del self._share_embed_names[embed_name]
      del self._share_embed_dims[embed_name]

    logging.info('shared embeddings[num=%d]' % len(self._share_embed_names))
    for embed_name in self._share_embed_names:
      logging.info('\t%s: share_num[%d], embed_dim[%d], partitioner[%s]' % (\
               embed_name, self._share_embed_names[embed_name],
               self._share_embed_dims[embed_name],
               self._share_embed_partitions[embed_name]))
    self._share_embed_columns = { embed_name : [] for embed_name in \
                                  self._share_embed_names }

    for config in self._feature_configs:
      assert isinstance(config, FeatureConfig)
      try:
        if config.feature_type == config.IdFeature:
          self.parse_id_feature(config)
        elif config.feature_type == config.TagFeature:
          self.parse_tag_feature(config)
        elif config.feature_type == config.RawFeature:
          self.parse_raw_feature(config)
        elif config.feature_type == config.ComboFeature:
          self.parse_combo_feature(config)
        elif config.feature_type == config.LookupFeature:
          self.parse_lookup_feature(config)
        elif config.feature_type == config.SequenceFeature:
          self.parse_sequence_feature(config)
        else:
          assert False, 'invalid feature type: %s' % config.feature_type
      except FeatureKeyError as ex:
        pass
    
    # for handling share embeded columns
    for embed_name in self._share_embed_names:
      share_embed_fcs = feature_column.shared_embedding_columns(
                          self._share_embed_columns[embed_name], 
                          self._share_embed_dims[embed_name], 
                          shared_embedding_collection_name=embed_name,
                          combiner='mean',
                          partitioner=self._share_embed_partitions[embed_name])
      self._share_embed_columns[embed_name] = share_embed_fcs

    for fc_name in self._deep_columns:
      fc = self._deep_columns[fc_name]
      if type(fc) == tuple:
        self._deep_columns[fc_name] = self._get_shared_embedding_column(fc)

  @property
  def wide_columns(self):
    return self._wide_columns

  @property
  def deep_columns(self):
    return self._deep_columns

  def is_wide(self, config):
    feature_name = None
    if config.HasField('feature_name'):
      feature_name = config.feature_name
    else:
      feature_name = config.input_names[0]
    if feature_name not in self._wide_deep_dict:
      raise FeatureKeyError(feature_name)
    return self._wide_deep_dict[feature_name] in [ WideOrDeep.WIDE, 
              WideOrDeep.WIDE_AND_DEEP ]

  def is_deep(self, config):
    feature_name = None
    if config.HasField('feature_name'):
      feature_name = config.feature_name
    else:
      feature_name = config.input_names[0]
    # DEEP or WIDE_AND_DEEP
    if feature_name not in self._wide_deep_dict:
      raise FeatureKeyError(feature_name)
    return self._wide_deep_dict[feature_name] in [ WideOrDeep.DEEP,
            WideOrDeep.WIDE_AND_DEEP ]

  def _build_partitioner(self, config):
    assert isinstance(config, FeatureConfig), "invalid config %s" % str(config)
    if config.HasField('max_partitions') and config.max_partitions > 1:
      return min_max_variable_partitioner(max_partitions=config.max_partitions)
    else:
      return None

  def parse_id_feature(self, config):
    feature_name = config.feature_name if config.HasField('feature_name') \
                     else config.input_names[0]
    hash_bucket_size = config.hash_bucket_size
    if hash_bucket_size > 0:
      fc = feature_column.categorical_column_with_hash_bucket(\
               config.input_names[0], hash_bucket_size=hash_bucket_size)
    elif config.vocab_list:
      fc = feature_column.categorical_column_with_vocabulary_list(\
               config.input_names[0], default_value=0,
               vocabulary_list=config.vocab_list)
    elif config.vocab_file:
      fc = feature_column.categorical_column_with_vocabulary_file(\
               config.input_names[0], default_value=0,
               vocabulary_file=config.vocab_file)
    else:
      fc = feature_column.categorical_column_with_identity(\
               config.input_names[0], config.num_buckets,
               default_value=0)

    if self.is_wide(config):
      self._wide_columns[feature_name] = feature_column.indicator_column(fc)

    if config.embedding_name in self._share_embed_names:
      fc = self._add_shared_embedding_column(config.embedding_name, fc)
    elif config.embedding_dim > 0:
      fc = feature_column.embedding_column(fc, config.embedding_dim,
                          partitioner=self._build_partitioner(config))
    else:
      fc = feature_column.indicator_column(fc)

    if self.is_deep(config): 
      self._deep_columns[feature_name] = fc

    return fc

  def parse_tag_feature(self, config):
    feature_name = config.feature_name if config.HasField('feature_name') \
                     else config.input_names[0]
    hash_bucket_size = config.hash_bucket_size
    if config.HasField('hash_bucket_size'):
      tag_fc = feature_column.categorical_column_with_hash_bucket(
                 config.input_names[0], hash_bucket_size, dtype=tf.string)
    else:
      tag_fc = feature_column.categorical_column_with_identity(
                 config.input_names[0], config.num_buckets,
                 default_value=0)
    
    if self.is_wide(config):
      self._wide_columns[feature_name] = feature_column.indicator_column(tag_fc)
  
    if len(config.input_names) > 1:
      tag_fc = feature_column.weighted_categorical_column(tag_fc,
                weight_feature_key=config.input_names[1], dtype=tf.float32)

    if config.embedding_name in self._share_embed_names:
      tag_fc = self._add_shared_embedding_column(config.embedding_name, tag_fc)
    elif config.embedding_dim > 0:
      tag_fc = feature_column.embedding_column(tag_fc, 
                  dimension=config.embedding_dim,
                  partitioner=self._build_partitioner(config))
    else:
      tag_fc = feature_column.indicator_column(tag_fc)

    if self.is_deep(config):
      self._deep_columns[feature_name] = tag_fc
  
    return tag_fc

  def parse_raw_feature(self, config):
    feature_name = config.feature_name if config.HasField('feature_name') \
                     else config.input_names[0]
    fc = feature_column.numeric_column(config.input_names[0])

    if self.is_wide(config):
      self._wide_columns[feature_name] = fc

    if config.boundaries:
      bounds = list(config.boundaries)
      bounds.sort()
      fc = feature_column.bucketized_column(fc, bounds)

      if self.is_wide(config):
        self._wide_columns[feature_name] = feature_column.indicator_column(fc)

      if config.embedding_name in self._share_embed_columns:
        fc = self._add_shared_embedding_column(config.embedding_name, fc)
      elif config.embedding_dim > 0:
        fc = feature_column.embedding_column(fc, config.embedding_dim,
                  partitioner=self._build_partitioner(config))
      else:
        fc = feature_column.indicator_column(fc)
   
    if self.is_deep(config):
      self._deep_columns[feature_name] = fc

  def _add_shared_embedding_column(self, embedding_name, fc):
    curr_id = len(self._share_embed_columns[embedding_name])
    self._share_embed_columns[embedding_name].append(fc)
    return (embedding_name, curr_id)

  def _get_shared_embedding_column(self, fc_handle):
    embed_name, embed_id = fc_handle
    return self._share_embed_columns[embed_name][embed_id]

  def parse_combo_feature(self, config):
    feature_name = config.feature_name if config.HasField('feature_name') \
                     else config.input_names[0]
    assert len(config.inputs) >= 2
    fc = feature_column.crossed_column(config.inputs,
              config.hash_bucket_size, hash_key=None) 

    if self.is_wide(config):
      self._wide_columns[feature_name] = feature_column.indicator_column(fc)

    if config.embedding_name in self._share_embed_columns:
      fc = self._add_shared_embedding_column(config.embedding_name, fc)
      # fc is now a handle
    elif config.embedding_dim > 0:
      fc = feature_column.embedding_column(fc, config.embedding_dim,
                  partitioner=self._build_partitioner(config))
    else:
      fc = feature_column.indicator_column(fc)

    if self.is_deep(config):
      self._deep_columns[feature_name] = fc

  def parse_lookup_feature(self, config):
    feature_name = config.feature_name if config.HasField('feature_name') \
                     else config.input_names[0]
    assert config.HasField('hash_bucket_size')
    hash_bucket_size = config.hash_bucket_size
    fc = feature_column.categorical_column_with_hash_bucket(
           feature_name, hash_bucket_size, dtype=tf.string)
    
    if self.is_wide(config):
      self._wide_columns[feature_name] = feature_column.indicator_column(fc)
  
    if config.embedding_name in self._share_embed_names:
      fc = self._add_shared_embedding_column(config.embedding_name, fc)
    elif config.embedding_dim > 0:
      fc = feature_column.embedding_column(fc, 
                  dimension=config.embedding_dim,
                  partitioner=self._build_partitioner(config))
    else:
      fc = feature_column.indicator_column(fc)

    if self.is_deep(config):
      self._deep_columns[feature_name] = fc
  
    return fc

  def parse_sequence_feature(self, config):
    feature_name = config.feature_name if config.HasField('feature_name') \
                       else config.input_names[0]
    assert config.HasField('hash_bucket_size')
    hash_bucket_size = config.hash_bucket_size
    fc = sequence_feature_column.sequence_categorical_column_with_hash_bucket(\
             feature_name, hash_bucket_size, dtype=tf.string) 
 
    assert config.embedding_dim > 0
    fc = feature_column.embedding_column(fc, dimension=config.embedding_dim,
                  partitioner=self._build_partitioner(config))
    self._deep_columns[feature_name] = fc

    return fc
