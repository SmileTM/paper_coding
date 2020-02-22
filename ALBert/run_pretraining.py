# -*- coding: utf-8 -*-
#
# File: run_pretraining.py
# Author: SmileTM
# Site: s-tm.cn
# Github: https://github.com/SmileTM
# Time: 02.22.2020
#


import tensorflow as tf
import modeling
import models
import os
import optimization
from absl import flags
from absl import app

FLAGS = flags.FLAGS

# bertconfig
flags.DEFINE_string("albert_config_file", "./albert_config.json",
                    "Albert configuration file to define core Albert layers.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained ALBERT model). "
    "The checkpoint from tfX_ckpt_converter.py")

# 导入 由creat_pretraining_data创建的tfrecord数据文件
flags.DEFINE_string(
    "input_file", "./tf_examples.tfrecor",
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
    tensorBoardCallback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(save_path, 'logs'), update_freq='batch')
    trainCallback = TrainCallback(num_train_steps=num_train_steps, save_path=save_path)
    saveCallback = tf.keras.callbacks.ModelCheckpoint(filepath=save_path,
                                                      monitor='loss',
                                                      save_best_only=True,
                                                      save_weights_only=True,
                                                      save_freq=5000)
    infoCallback = tf.keras.callbacks.CSVLogger(os.path.join(save_path, 'pretrain.log'))

    callbacks = [trainCallback, saveCallback, infoCallback, tensorBoardCallback]
    return callbacks


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
            'seq_relationship': parse_example['next_sentence_labels']

        }
        y = parse_example['masked_lm_weights']

        return (x, y)

    raw_dataset = tf.data.TFRecordDataset(file_path)

    dataset = raw_dataset.map(parse_function)
    dataset = dataset.batch(batch_size=train_batch_size, drop_remainder=True)
    return dataset


def main(_):
    dataset = load_data(train_batch_size=FLAGS.train_batch_size,
                        file_path=FLAGS.input_file,
                        max_seq_length=FLAGS.max_seq_length,
                        max_predictions_per_seq=FLAGS.max_predictions_per_seq)

    callbacks = get_callbasks(num_train_steps=FLAGS.train_batch_size, save_path=FLAGS.output_dir)

    config = modeling.AlbertConfig.from_json_file(FLAGS.albert_config_file)

    pretraining_model = models.getPretrainingModel(config=config,
                                                   max_predictions_per_seq=FLAGS.max_predictions_per_seq,
                                                   max_seq_length=FLAGS.max_seq_length)
    pretraining_model.summary()

    if FLAGS.init_checkpoint:
        # 如果 init_checkpoint 存在 这用init_checkpoint 进行初始化， 相当于导入权重
        pretraining_model.load_weights(FLAGS.init_checkpoint)

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
