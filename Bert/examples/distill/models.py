# -*- coding: utf-8 -*-
#
# File: models.py
# Author: SmileTM
# Site: s-tm.cn
# Github: https://github.com/SmileTM
# Time: 02.20.2020
#

import tensorflow as tf
import modeling


def classifile_model(config, max_seq_length):
    bert_model = modeling.BertModel(config, name="bert")

    input_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), name='input_ids', dtype=tf.int32)
    input_mask = tf.keras.layers.Input(
        shape=(max_seq_length,), name='input_mask', dtype=tf.int32)
    segment_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), name='segment_ids', dtype=tf.int32)

    pooled_output, _ = bert_model((input_ids, input_mask, segment_ids))
    out = tf.keras.layers.Dropout(0.1)(pooled_output)

    out = tf.keras.layers.Dense(config.hidden_size)(out)
    logits = tf.keras.layers.Dense(2)(out)

    inputs = [input_ids, input_mask, segment_ids]

    model = tf.keras.Model(inputs=inputs, outputs=logits)
    return model


class ClassifilerModel(tf.keras.Model):
    def __init__(self, config, num_class, teacher_hidden=768, **kwargs):
        super(ClassifilerModel, self).__init__(**kwargs)
        self.config = config
        self.num_class = num_class
        self.teacher_hidden = teacher_hidden

    def build(self, input_shape):
        self.bert_model = modeling.BertModel(self.config, name='bert')
        self.drop = tf.keras.layers.Dropout(0.1)
        self.drop_dense = tf.keras.layers.Dense(self.config.hidden_size, name='drop_dense')
        self.out_dense = tf.keras.layers.Dense(self.num_class, name='out_dense')
        self.distill_dense = tf.keras.layers.Dense(self.teacher_hidden, name='distill_dense')

    def call(self, inputs, mode='train'):
        (input_ids, input_mask, segment_ids) = inputs
        if mode == 'distill':
            input_tensor, attns_output, pooled_output = self.bert_model((input_ids, input_mask, segment_ids), mode=mode)
            input_tensor = self.distill_dense(input_tensor)
            for layer_index, attn_out in enumerate(attns_output):
                attns_output[layer_index] = self.distill_dense(attn_out)

            drop_out = self.drop(pooled_output)
            drop_dens_out = self.drop_dense(drop_out)
            logits = self.out_dense(drop_dens_out)

            return input_tensor, attns_output, logits
        else:
            pooled_output, sequence_output = self.bert_model((input_ids, input_mask, segment_ids))
            drop_out = self.drop(pooled_output)
            drop_dens_out = self.drop_dense(drop_out)
            logits = self.out_dense(drop_dens_out)
            return logits


class BaseModel(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super(BaseModel, self).__init__(**kwargs)
        self.config = config

    def build(self, input_shape):
        self.bert_model = modeling.BertModel(self.config, name='bert')

    def call(self, inputs, mode='train'):
        input_ids, input_mask, segment_ids = inputs
        pooled_output, sequence_output = self.bert_model((input_ids, input_mask, segment_ids))
        return pooled_output, sequence_output
