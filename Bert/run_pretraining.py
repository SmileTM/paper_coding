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
import os
import optimization
from absl import flags
from absl import app

FLAGS = flags.FLAGS

# bertconfig 在main中定义
#

# 导入 由creat_pretraining_data创建的tfrecord数据文件
flags.DEFINE_string(
    "input_file", "./tf_examples.tfrecord",
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "output_dir", "./out",
    "The output directory where the model checkpoints will be written.")
# Model training specific flags.
flags.DEFINE_integer(
    'max_seq_length', 512,
    'The maximum total input sequence length after WordPiece tokenization. '
    'Sequences longer than this will be truncated, and sequences shorter '
    'than this will be padded.')
flags.DEFINE_integer('max_predictions_per_seq', 20,
                     'Maximum predictions per sequence_output.')
flags.DEFINE_integer('train_batch_size', 32, 'Total batch size for training.')
flags.DEFINE_integer('num_steps_per_epoch', 1000,
                     'Total number of training steps to run per epoch.')
flags.DEFINE_float('warmup_steps', 10000,
                   'Warmup steps for Adam weight decay optimizer.')

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")


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

        per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=-1)
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


class TrainCallback(tf.keras.callbacks.Callback):
    def __init__(self, num_train_steps, save_path):
        self.num_train_steps = num_train_steps
        self.step = 0
        self.save_path = save_path

    def on_batch_end(self, batch, logs=None):
        if self.step < self.num_train_steps:
            self.step += 1
        else:
            self.model.save_weights(os.path.join(self.save_path, f'step-{self.step}-ckpt'))
            self.model.stop_training = True


def get_callbasks(num_train_steps, save_path):
    TensorBoardCallback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(save_path, 'logs'), update_freq='batch')
    TrainCallback(num_train_steps=num_train_steps, save_path=save_path)
    SaveCallback = tf.keras.callbacks.ModelCheckpoint(filepath=save_path,
                                                      monitor='loss',
                                                      save_best_only=True,
                                                      save_weights_only=True,
                                                      save_freq=5000)
    InfoCallback = tf.keras.callbacks.CSVLogger(os.path.join(save_path, 'pretrain.log'))

    callbacks = [TrainCallback, SaveCallback, InfoCallback]
    return callbacks


def getPretrainingModel(config, max_seq_length, max_predictions_per_seq):
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

    bert_model = modeling.BertModel(config)
    pooled_output, sequence_output = bert_model((input_ids, input_mask, segment_ids))

    mask_label_loss = Pretraining_mask_label_loss_layer(source_network=bert_model,
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


def load_data(file_path, train_batch_size, max_seq_length, max_predictions_per_seq):
    def parse_function(example):
        # 这里的dtype 类型只有 float32, int64, string
        feature_description = {
            "input_ids":
                tf.io.FixedLenFeature([max_seq_length], tf.int64),
            "input_mask":
                tf.io.FixedLenFeature([max_seq_length], tf.int64),
            "segment_ids":
                tf.io.FixedLenFeature([max_seq_length], tf.int64),
            "masked_lm_positions":
                tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_ids":
                tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_weights":
                tf.io.FixedLenFeature([max_predictions_per_seq], tf.float32),
            "next_sentence_labels":
                tf.io.FixedLenFeature([1], tf.int64),
        }

        parse_example = tf.io.parse_single_example(example, feature_description)

        x = {
            'input_ids': parse_example['input_ids'],
            'input_mask': parse_example['input_mask'],
            'segment_ids': parse_example['segment_ids'],
            'masked_lm_positions': parse_example['masked_lm_positions'],
            'masked_lm_ids': parse_example['masked_lm_ids'],
            'masked_lm_weights': parse_example['masked_lm_weights'],
            'next_sentence_labels': parse_example['next_sentence_labels']

        }
        y = parse_example['masked_lm_weights']

        return (x, y)

    raw_dataset = tf.data.TFRecordDataset(file_path)

    dataset = raw_dataset.map(parse_function)
    dataset = dataset.batch(batch_size=train_batch_size, drop_remainder=True)
    return dataset


def load():
    config = modeling.BertConfig()
    mode = getPretrainingModel()
    mode.load_weights('./out/allmodel-ckpt')


def main(_):
    dataset = load_data(train_batch_size=FLAGS.train_batch_size,
                        file_path=FLAGS.input_file,
                        max_seq_length=FLAGS.max_seq_length,
                        max_predictions_per_seq=FLAGS.max_predictions_per_seq)
    callbacks = get_callbasks(num_train_steps=FLAGS.train_batch_size, save_path=FLAGS.output_dir)

    config = modeling.BertConfig(vocab_size=30522,
                                 n_head=12,
                                 d_hidden=768,
                                 num_hidden_layers=12,
                                 d_intermediate=3072,
                                 max_position_embedding=512,
                                 type_vocab_size=16,
                                 hidden_act='gelu',
                                 initializer_range=0.02,
                                 attention_dropout_rate=0.0,
                                 hidden_dropout_rate=0.0)

    pretraining_model = getPretrainingModel(config=config, max_predictions_per_seq=FLAGS.max_predictions_per_seq,
                                            max_seq_length=FLAGS.max_seq_length)
    pretraining_model.summary()

    loss = lambda y_true, y_pred: y_pred
    optimizer = optimization.create_optimizer(init_lr=FLAGS.learning_rate,
                                              num_train_steps=FLAGS.train_batch_size,
                                              num_warmup_steps=FLAGS.warmup_steps)

    pretraining_model.compile(optimizer=optimizer, loss=loss)
    pretraining_model.fit(dataset, epochs=100, callbacks=callbacks)


if __name__ == '__main__':
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("max_seq_length")
    flags.mark_flag_as_required("max_predictions_per_seq")

    app.run(main)
