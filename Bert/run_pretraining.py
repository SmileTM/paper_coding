# -*- coding: utf-8 -*-
#
# File: run_pretraining.py
# Author: SmileTM
# Site: s-tm.cn
# Github: https://github.com/SmileTM
# Time: 12.02.2020
#

import tensorflow as tf
import modeling
import utils


class Pretraining_mask_label_loss_layer(tf.keras.layers.Layer):
    def __init__(self, source_network, **kwargs):
        super(Pretraining_mask_label_loss_layer, self).__init__()
        self.config = source_network.config
        self.embedding_table = source_network.embedding_processor.embedding_word_ids.embeddings

    def build(self, input_shape):
        self.dense = tf.keras.layers.Dense(units=self.config.d_hidden,
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

        per_example_loss = -tf.reduce_sum(log_probs*one_hot_labels, axis=-1)
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator
        return loss

    def gather_indexes(self, sequence_tensor, positions):
        sequence_tensor_shape = sequence_tensor.shape
        batch_size = sequence_tensor_shape[0]
        seq_length = sequence_tensor_shape[1]
        width = sequence_tensor_shape[2]

        flat_offset = tf.reshape(tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
        flat_position = tf.reshape(positions + flat_offset, [-1])

        flat_sequencea_tensor = tf.reshape(sequence_tensor, (-1, width))

        output_tensor = tf.gather(flat_sequencea_tensor, flat_position)
        return output_tensor


class Pretraining_next_sentence_loss_layer(tf.keras.layers.Layer):
    def __init__(self, source_network, **kwargs):
        self.config = source_network.config
        super(Pretraining_next_sentence_loss_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dens = tf.keras.layers.Dense(units=2,
                                          kernel_initializer=utils.get_initializer(self.config.initializer_range))
        super(Pretraining_next_sentence_loss_layer, self).build(input_shape)

    def call(self, inputs):
        (input_tensor, labels) = inputs
        logits = self.dens(input_tensor)

        log_probs = tf.nn.log_softmax(logits, axis=-1)

        labels = tf.reshape(labels, shape=[-1])
        one_hor_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)

        per_example_loss = - tf.reduce_sum(log_probs * one_hor_labels, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return loss


def getPretrainingModel():
    config = modeling.BertConfig()
    seq_length = 512
    max_predictions_per_seq = 20
    input_word_ids = tf.keras.layers.Input(
        shape=(seq_length,), name='input_word_ids', dtype=tf.int32)
    input_mask = tf.keras.layers.Input(
        shape=(seq_length,), name='input_mask', dtype=tf.int32)
    input_type_ids = tf.keras.layers.Input(
        shape=(seq_length,), name='input_type_ids', dtype=tf.int32)
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

    bert_model = modeling.BertModel(config)
    pooled_output, sequence_output = bert_model((input_word_ids, input_mask, input_type_ids))

    mask_label_loss = Pretraining_mask_label_loss_layer(source_network=bert_model, name='mask_label_loss')((sequence_output,
                                                                                          masked_lm_positions,
                                                                                          masked_lm_ids,
                                                                                          masked_lm_weights))
    next_sentence_loss = Pretraining_next_sentence_loss_layer(source_network=bert_model, name='next_sentence_loss')((pooled_output,
                                                                                                next_sentence_labels))

    inputs = [input_word_ids, input_mask, input_type_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights,
              next_sentence_labels]
    outputs = mask_label_loss+next_sentence_loss
    pretraining_model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    return pretraining_model



def train():
    pretrainingModel=getPretrainingModel()
    pretrainingModel.summary()
    print(pretrainingModel.layers)
    print(len(pretrainingModel.layers))
    pretrainingModel.submodules

    # myloss = lambda y_true,y_pred:y_pred
    # pretrainingModel.compile(optimizer=tf.keras.optimizers.Adam(),
    #                          loss=myloss)

if __name__ == '__main__':
    train()
