# -*- coding: utf-8 -*-
#
# File: train_student.py
# Author: SmileTM
# Site: s-tm.cn
# Github: https://github.com/SmileTM
# Time: 09.09.2020
#
import tensorflow as tf
import numpy as np
from Bert import tokenization
import modeling, models
from data_process import get_dataset
from tqdm import tqdm

import os

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

max_seq_length = 512
BATCH_SIZE = 4
initial_lr = 5e-5
EPOCHS = 10

vocab_path = 'vocab.txt'

tokenizer = tokenization.FullTokenizer(vocab_path, do_lower_case=True)

train_dataset, train_label, test_dataset, test_label = get_dataset(tokenizer, max_seq_length)
trainDataSet = tf.data.Dataset.from_tensor_slices((train_dataset, train_label)).shuffle(30000).batch(BATCH_SIZE,
                                                                                                     drop_remainder=True).prefetch(
    tf.data.experimental.AUTOTUNE)
testDataset = tf.data.Dataset.from_tensor_slices((test_dataset, test_label)).batch(BATCH_SIZE).prefetch(
    tf.data.experimental.AUTOTUNE)

# steps_per_epoch = train_label.shape[0] // BATCH_SIZE
# all_steps = EPOCHS * steps_per_epoch
# warmup_steps = int(all_steps * 0.1)

teacher_config = modeling.BertConfig.from_json_file('teacher_config.json')
teacher_model = models.ClassifilerModel(teacher_config, 2)
student_config = modeling.BertConfig.from_json_file('student_config.json')
student_model = models.ClassifilerModel(student_config, 2, teacher_hidden=teacher_config.hidden_size,
                                        name='student_model')

optimizer = tf.keras.optimizers.Adam(5e-5)
# body_trained = False
body_trained = True
temperature = 3

teacher_ck = tf.train.Checkpoint(model=teacher_model)
teacher_ck.restore(tf.train.latest_checkpoint('teacher_model'))
student_ck = tf.train.Checkpoint(model=student_model)
if body_trained:
    student_ck.restore(tf.train.latest_checkpoint('student_model'))
for epoch in range(EPOCHS):
    if not body_trained:
        # distill body
        total_loss = tf.keras.metrics.Mean()
        for step, dataset in tqdm(enumerate(trainDataSet)):
            data, label = dataset
            with tf.GradientTape() as tape:
                t_input_tensor, t_attns_output, t_logits = teacher_model(
                    (data['input_ids'], data['input_mask'], data['segment_ids']), mode='distill')
                s_input_tensor, s_attns_output, s_logits = student_model(
                    (data['input_ids'], data['input_mask'], data['segment_ids']), mode='distill')
                em_loss = 0
                attn_loss = 0
                t_input_tensor = tf.reshape(t_input_tensor, -1)
                s_input_tensor = tf.reshape(s_input_tensor, -1)
                em_loss = tf.keras.losses.MSE(t_input_tensor, s_input_tensor)

                for layer_index, attn_out in enumerate(s_attns_output, 1):
                    s_attn_out = tf.reshape(attn_out, -1)
                    t_attn_out = tf.reshape(
                        t_attns_output[(layer_index * int(len(t_attns_output) / len(s_attns_output))) - 1], -1)
                    attn_loss += tf.keras.losses.MSE(t_attn_out, s_attn_out)

                loss = em_loss + attn_loss
                grads = tape.gradient(loss, student_model.trainable_variables)
                optimizer.apply_gradients(zip(grads, student_model.trainable_variables))
                total_loss(loss)
                if step % 100 == 0:
                    print("distill body epoch{}  setp {}  loss: {}".format(epoch, step, total_loss.result()))
        print("distill body epoch{} loss: {}".format(epoch, total_loss.result()))

        student_ck.save('student_model/model')

    else:
        # distill pred
        total_loss = tf.keras.metrics.Mean()
        train_acc = tf.keras.metrics.Accuracy()
        test_acc = tf.keras.metrics.Accuracy()
        for step, dataset in tqdm(enumerate(trainDataSet)):
            data, label = dataset
            with tf.GradientTape() as tape:
                t_logits = teacher_model((data['input_ids'], data['input_mask'], data['segment_ids']))
                s_logits = student_model((data['input_ids'], data['input_mask'], data['segment_ids']))

                losses = -tf.nn.softmax(t_logits / temperature) * tf.nn.log_softmax(s_logits / temperature)
                losses = tf.reduce_sum(losses, axis=-1)
                loss = tf.reduce_mean(losses)

            grads = tape.gradient(loss, student_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, student_model.trainable_variables))
            predictions = tf.argmax(s_logits, axis=-1, output_type=tf.int32)
            total_loss(loss)
            train_acc(label, predictions)
            if step % 100 == 0:
                print("distill pred epoch{}  setp {} acc: {:.3%}  loss: {}".format(epoch, step, train_acc.result(),
                                                                                   total_loss.result()))

        print('evaluating')
        for dataset in tqdm(testDataset):
            data, label = dataset
            s_logits = student_model((data['input_ids'], data['input_mask'], data['segment_ids']))
            predictions = tf.argmax(s_logits, axis=-1, output_type=tf.int32)
            test_acc(label, predictions)

        student_ck.save('student_model/model')

        print(
            f'distill pred, epoch {epoch}, train loss: {total_loss.result()}, train acc: {train_acc.result()}, test acc: {test_acc.result()}')
