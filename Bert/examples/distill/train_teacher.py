# -*- coding: utf-8 -*-
#
# File: train_teacher.py
# Author: SmileTM
# Site: s-tm.cn
# Github: https://github.com/SmileTM
# Time: 09.10.2020
#

import tensorflow as tf
import numpy as np
from Bert import tokenization
import modeling, models
from data_process import get_dataset
from tqdm import tqdm
import optimization
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
max_seq_length = 512
BATCH_SIZE = 8
initial_lr = 5e-5
EPOCHS = 5
vocab_path = './vocab.txt'

tokenizer = tokenization.FullTokenizer(vocab_path, do_lower_case=True)

train_dataset, train_label, test_dataset, test_label = get_dataset(tokenizer, max_seq_length)
trainDataset = tf.data.Dataset.from_tensor_slices((train_dataset, train_label)).shuffle(30000).batch(BATCH_SIZE,
                                                                                                     drop_remainder=True).prefetch(
    tf.data.experimental.AUTOTUNE)

testDataset = tf.data.Dataset.from_tensor_slices((test_dataset, test_label)).batch(BATCH_SIZE).prefetch(
    tf.data.experimental.AUTOTUNE)

steps_per_epoch = train_label.shape[0] // BATCH_SIZE
all_steps = EPOCHS * steps_per_epoch
warmup_steps = int(all_steps * 0.1)

teacher_config = modeling.BertConfig.from_json_file('teacher_config.json')

teacher_model = models.ClassifilerModel(teacher_config, 2)
optimizer = optimization.create_optimizer(initial_lr,
                                          all_steps,
                                          warmup_steps,
                                          )
checkpoint = tf.train.Checkpoint(model=teacher_model)
checkpoint.restore('distill/ck_savedmodel/base_model.ckpt')


@tf.function
def train(model, data, label):
    with tf.GradientTape() as tape:
        logits = model((data['input_ids'], data['input_mask'], data['segment_ids']))
        loss = tf.keras.losses.sparse_categorical_crossentropy(label, logits, from_logits=True)
    grads = tape.gradient(loss, model.trainable_variables)
    return loss, grads, logits


for epoch in range(EPOCHS):
    print(f'trainning {epoch}')
    total_loss = tf.keras.metrics.Mean()
    train_accuracy = tf.keras.metrics.Accuracy()
    for step, (data, label) in tqdm(enumerate(trainDataset)):
        loss, grads, logits = train(teacher_model, data, label)
        optimizer.apply_gradients(zip(grads, teacher_model.trainable_variables))
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        train_accuracy(label, predictions)
        total_loss(loss)
        if step % 100 == 0:
            print("train epoch{}  setp {} acc: {:.3%}  loss: {}".format(epoch, step, train_accuracy.result(),
                                                                        total_loss.result()))

    print("train set accuracy: {:.3%}  loss: {}".format(train_accuracy.result(), total_loss.result()))

    print('evaluating')
    eval_accuracy = tf.keras.metrics.Accuracy()
    for data, label in tqdm(testDataset):
        logits = teacher_model((data['input_ids'], data['input_mask'], data['segment_ids']))
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        eval_accuracy(label, predictions)

    print("eval set accuracy: {:.3%}".format(eval_accuracy.result()))
    checkpoint.save('teacher_model/model')
