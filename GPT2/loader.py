# -*- coding: utf-8 -*-
#
# File: read_tfrecord.py
# Author: SmileTM
# Site: s-tm.cn
# Github: https://github.com/SmileTM
# Time: 11.11.2020
#
import tensorflow as tf
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE


class DataSet(object):

    def __init__(self, tfrecord_dir, batch_size):
        self.tfrecord_dir = tfrecord_dir
        self.batch_size = batch_size

    def _parse_function(self, example):
        # 这里的dtype 类型只有 float32, int64, string
        feature_description = {
            'token_ids': tf.io.FixedLenFeature(shape=(512,), dtype=tf.int64),
            'token_type_ids': tf.io.FixedLenFeature(shape=(512,), dtype=tf.int64),
        }

        parse_example = tf.io.parse_single_example(example, feature_description)
        parse_example['token_ids'] = tf.cast(parse_example['token_ids'], tf.int32)
        parse_example['token_type_ids'] = tf.cast(parse_example['token_type_ids'], tf.int32)
        return parse_example

    def train_dataset(self):
        data = tf.data.TFRecordDataset(os.path.join(self.tfrecord_dir, 'train.tfrecord'), num_parallel_reads=AUTOTUNE)
        data = data.map(self._parse_function, num_parallel_calls=AUTOTUNE)
        data = data.shuffle(1000000).batch(self.batch_size).prefetch(AUTOTUNE)
        return data

    def eval_dataset(self):
        data = tf.data.TFRecordDataset(os.path.join(self.tfrecord_dir, 'valid.tfrecord'), num_parallel_reads=AUTOTUNE)
        data = data.map(self._parse_function, num_parallel_calls=AUTOTUNE)
        data = data.batch(self.batch_size).prefetch(AUTOTUNE)
        return data

    def test_dataset(self):
        data = tf.data.TFRecordDataset(os.path.join(self.tfrecord_dir, 'test.tfrecord'), num_parallel_reads=AUTOTUNE)
        data = data.map(self._parse_function, num_parallel_calls=AUTOTUNE)
        data = data.batch(self.batch_size).prefetch(AUTOTUNE)
        return data
