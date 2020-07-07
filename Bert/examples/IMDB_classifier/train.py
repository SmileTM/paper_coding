# -*- coding: utf-8 -*-
#
# File: train.py
# Author: SmileTM
# Site: s-tm.cn
# Github: https://github.com/SmileTM
# Time: 07.06.2020
#
import tensorflow as tf
from Bert import tokenization
from data_process import get_dataset
from Bert import modeling
from Bert import optimization

max_seq_length = 512
BATCH_SIZE = 16
initial_lr = 5e-5
EPOCHS = 3
vocab_path = './uncased_L-12_H-768_A-12/vocab.txt'
bert_config_path = './uncased_L-12_H-768_A-12/bert_config.json'

tokenizer = tokenization.FullTokenizer(vocab_path, do_lower_case=True)

train_dataset, train_label, test_dataset, test_label = get_dataset(tokenizer, max_seq_length)

steps_per_epoch = train_label.shape[0] // BATCH_SIZE
all_steps = EPOCHS * steps_per_epoch
warmup_steps = int(all_steps * 0.1)

config = modeling.BertConfig.from_json_file(bert_config_path)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    bert_model = modeling.BertModel(config, name="bert")

    input_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), name='input_ids', dtype=tf.int32)
    input_mask = tf.keras.layers.Input(
        shape=(max_seq_length,), name='input_mask', dtype=tf.int32)
    segment_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), name='segment_ids', dtype=tf.int32)

    pooled_output, _ = bert_model((input_ids, input_mask, segment_ids))
    out = tf.keras.layers.Dropout(0.1)(pooled_output)

    out = tf.keras.layers.Dense(config.hidden_size)(pooled_output)
    logits = tf.keras.layers.Dense(2)(out)

    inputs = [input_ids, input_mask, segment_ids]

    model = tf.keras.Model(inputs=inputs, outputs=logits)

    model.load_weights('./out_new/bert_model.ckpt')

    optimizer = optimization.create_optimizer(initial_lr,
                                              all_steps,
                                              warmup_steps,
                                              )
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['acc'])

    model.fit(train_dataset, train_label, steps_per_epoch=steps_per_epoch, epochs=EPOCHS, batch_size=BATCH_SIZE,
              shuffle=True,
              validation_data=(test_dataset, test_label))
