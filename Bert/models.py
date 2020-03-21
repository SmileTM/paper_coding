# -*- coding: utf-8 -*-
#
# File: models.py
# Author: SmileTM
# Site: s-tm.cn
# Github: https://github.com/SmileTM
# Time: 02.20.2020
#

import tensorflow as tf
import utils
import modeling


class Pretraining_mask_label_loss_layer(tf.keras.layers.Layer):
    def __init__(self, source_network, every_device_batch_size, **kwargs):
        super(Pretraining_mask_label_loss_layer, self).__init__(**kwargs)
        self.source_network = source_network
        self.batch_size = every_device_batch_size
        self.config = self.source_network.config
        self.embedding_table = self.source_network.embedding_processor.embedding_word_ids.embeddings

    def build(self, input_shape):
        self.dense = tf.keras.layers.Dense(units=self.config.hidden_size,
                                           activation=utils.get_activation(self.config.hidden_act),
                                           kernel_initializer=utils.get_initializer(self.config.initializer_range))

        self.layer_norm = tf.keras.layers.LayerNormalization(name="layer_norm",
                                                             axis=-1,
                                                             epsilon=1e-12,
                                                             dtype=tf.float32)
        self.output_bias = self.add_weight(name='output_bias',
                                           shape=[self.config.vocab_size],
                                           initializer=tf.keras.initializers.zeros())

        super(Pretraining_mask_label_loss_layer, self).build(input_shape)

    def call(self, inputs):
        (input_tensor, positions, label_ids, label_weights) = inputs
        input_tensor = self.gather_indexes(input_tensor, positions)
        input_tensor = self.dense(input_tensor)
        input_tensor = self.layer_norm(input_tensor)
        label_weights = tf.cast(label_weights, tf.float32)

        logits = tf.matmul(input_tensor, self.embedding_table, transpose_b=True)
        logits = tf.nn.bias_add(logits, self.output_bias)

        log_probs = tf.nn.log_softmax(logits, axis=-1)
        label_ids = tf.reshape(label_ids, [-1])
        label_weights = tf.reshape(label_weights, [-1])

        one_hot_labels = tf.one_hot(label_ids, depth=self.config.vocab_size, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=-1)
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator

        return loss

    def gather_indexes(self, sequence_tensor, positions):
        sequence_tensor_shape = sequence_tensor.shape
        batch_size = self.batch_size
        seq_length = sequence_tensor_shape[1]
        width = sequence_tensor_shape[2]

        flat_offset = tf.reshape(tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
        flat_position = tf.reshape(positions + flat_offset, [-1])

        flat_sequencea_tensor = tf.reshape(sequence_tensor, (-1, width))

        output_tensor = tf.gather(flat_sequencea_tensor, flat_position)
        return output_tensor

    def get_config(self):
        config = super(Pretraining_mask_label_loss_layer, self).get_config()
        config.update({
            "source_network": self.source_network,
            "batche_size": self.batch_size
        })
        return config


class Pretraining_next_sentence_loss_layer(tf.keras.layers.Layer):
    def __init__(self, source_network, **kwargs):
        super(Pretraining_next_sentence_loss_layer, self).__init__(**kwargs)
        self.source_network = source_network
        self.config = self.source_network.config

    def build(self, input_shape):
        self.dense = tf.keras.layers.Dense(units=2,
                                           kernel_initializer=utils.get_initializer(self.config.initializer_range))
        super(Pretraining_next_sentence_loss_layer, self).build(input_shape)

    def call(self, inputs):
        (input_tensor, labels) = inputs
        logits = self.dense(input_tensor)

        log_probs = tf.nn.log_softmax(logits, axis=-1)

        labels = tf.reshape(labels, shape=[-1])
        one_hor_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)

        per_example_loss = - tf.reduce_sum(log_probs * one_hor_labels, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return loss

    def get_config(self):
        config = super(Pretraining_next_sentence_loss_layer, self).get_config()
        config.update({
            "source_network": self.source_network
        })
        return config


def get_base_model(config, max_seq_length=512):
    config = config
    seq_length = max_seq_length
    input_ids = tf.keras.layers.Input(
        shape=(seq_length,), name='input_ids', dtype=tf.int32)
    input_mask = tf.keras.layers.Input(
        shape=(seq_length,), name='input_mask', dtype=tf.int32)
    segment_ids = tf.keras.layers.Input(
        shape=(seq_length,), name='segment_ids', dtype=tf.int32)

    bert_model = modeling.BertModel(config, name="bert")
    pooled_output, sequence_output = bert_model((input_ids, input_mask, segment_ids))

    inputs = [input_ids, input_mask, segment_ids]
    outputs = [pooled_output, sequence_output]

    base_model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return base_model


def getPretrainingModel(config, max_seq_length, every_device_batch_size=1, max_predictions_per_seq=20):
    config = config
    seq_length = max_seq_length
    max_predictions_per_seq = max_predictions_per_seq
    input_ids = tf.keras.layers.Input(
        shape=(seq_length,), name='input_ids', dtype=tf.int32)
    input_mask = tf.keras.layers.Input(
        shape=(seq_length,), name='input_mask', dtype=tf.int32)
    segment_ids = tf.keras.layers.Input(
        shape=(seq_length,), name='segment_ids', dtype=tf.int32)
    masked_lm_positions = tf.keras.layers.Input(
        shape=(max_predictions_per_seq,),
        name='masked_lm_positions',
        dtype=tf.int32)
    masked_lm_ids = tf.keras.layers.Input(
        shape=(max_predictions_per_seq,), name='masked_lm_ids', dtype=tf.int32)
    masked_lm_weights = tf.keras.layers.Input(
        shape=(max_predictions_per_seq,),
        name='masked_lm_weights',
        dtype=tf.int32)
    next_sentence_labels = tf.keras.layers.Input(
        shape=(1,), name='next_sentence_labels', dtype=tf.int32)

    bert_model = modeling.BertModel(config, name="bert")
    pooled_output, sequence_output = bert_model((input_ids, input_mask, segment_ids))

    mask_label_loss = Pretraining_mask_label_loss_layer(source_network=bert_model,
                                                        every_device_batch_size=every_device_batch_size,
                                                        name='mask_label_loss')((sequence_output,
                                                                                 masked_lm_positions,
                                                                                 masked_lm_ids,
                                                                                 masked_lm_weights))

    next_sentence_loss = Pretraining_next_sentence_loss_layer(source_network=bert_model,
                                                              name='next_sentence_loss')((pooled_output,
                                                                                          next_sentence_labels))

    inputs = [input_ids, input_mask, segment_ids, masked_lm_positions,
              masked_lm_ids, masked_lm_weights, next_sentence_labels]

    # 将经过reduce_sum得到的无维度数值，添加一个维度，避免传入loss keras计算出错
    outputs = tf.expand_dims(mask_label_loss + next_sentence_loss, axis=0)

    pretraining_model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    return pretraining_model
