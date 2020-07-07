# -*- coding: utf-8 -*-
#
# File: train_hub.py
# Author: SmileTM
# Site: s-tm.cn
# Github: https://github.com/SmileTM
# Time: 07.07.2020
#

# -*- coding: utf-8 -*-
#
# File: train.py
# Author: SmileTM
# Site: s-tm.cn
# Github: https://github.com/SmileTM
# Time: 07.06.2020
#

import tensorflow as tf

from data_process import get_hub_dataset
from Bert import tokenization
from Bert import optimization
import tensorflow_hub as hub

max_seq_length = 512
vocab_path = './uncased_L-12_H-768_A-12/vocab.txt'

tokenizer = tokenization.FullTokenizer(vocab_path, do_lower_case=True)

train_dataset, train_label, test_dataset, test_label = get_hub_dataset(tokenizer, max_seq_length)

input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_word_ids")
input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                   name="input_mask")
segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                    name="segment_ids")
bert_layer = hub.KerasLayer("https://hub.tensorflow.google.cn/tensorflow/bert_en_uncased_L-12_H-768_A-12/2",
                            trainable=True)
pooled_output, _ = bert_layer([input_word_ids, input_mask, segment_ids])
out = tf.keras.layers.Dropout(0.1)(pooled_output)
logits = tf.keras.layers.Dense(2)(out)

inputs = [input_word_ids, input_mask, segment_ids]

model = tf.keras.Model(inputs=inputs, outputs=logits)

warmup_steps = int(3 * 6250 * 0.1)
initial_lr = 5e-5
optimizer = optimization.create_optimizer(initial_lr,
                                          6250 * 3,
                                          warmup_steps,
                                          )
# model.optimizer = optimizer

model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['acc'])

model.fit(train_dataset, train_label, steps_per_epoch=6250, epochs=3, batch_size=4, shuffle=True,
          validation_data=(test_dataset, test_label))
