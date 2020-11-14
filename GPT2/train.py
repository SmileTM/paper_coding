# -*- coding: utf-8 -*-
#
# File: train.py
# Author: SmileTM
# Site: s-tm.cn
# Github: https://github.com/SmileTM
# Time: 11.10.2020
#
import modeling
import optimization
from loader import DataSet
import tensorflow as tf
import os
from tqdm import tqdm

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)

BATCH_SIZE = 64
EPOCHS = 10
total_step = 1250
every_step_save = 10
ckpt_output_dir = './model'

dataset = DataSet('./data', BATCH_SIZE)
train_dataset = dataset.train_dataset()
eval_dataset = dataset.eval_dataset().take(1)
test_dataset = dataset.test_dataset()

strategy = tf.distribute.MirroredStrategy()
# 数据分发
dist_train_dataset = strategy.experimental_distribute_dataset(train_dataset)
dist_eval_dataset = strategy.experimental_distribute_dataset(eval_dataset)
dist_test_dataset = strategy.experimental_distribute_dataset(test_dataset)

with strategy.scope():
    # 模型 优化器 定义
    config = modeling.GPT2Config()
    model = modeling.GPT2Model(config)
    # optimizer = optimization.create_optimizer(init_lr=5e-5,
    #                                           total_step=total_step,
    #                                           num_warmup_steps=int(total_step*0.1))
    optimizer = tf.keras.optimizers.Adam()
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(checkpoint,
                                         directory=ckpt_output_dir,
                                         max_to_keep=3,
                                         step_counter=optimizer.iterations,
                                         checkpoint_interval=every_step_save)

    # loss 记录器
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    eval_loss = tf.keras.metrics.Mean(name="eval_loss")


    def computer_loss(y_true, y_pred):
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
        loss = tf.reduce_mean(loss)
        return loss


@tf.function
def train_step(dist_data):
    with strategy.scope():
        def train_fn(input_data):
            token_ids, token_types = input_data['token_ids'], input_data['token_type_ids']
            with tf.GradientTape() as tape:
                logits = model((token_ids, token_types))
                loss = computer_loss(token_ids[:, 1:], logits[:, :-1])
                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
            return loss
    per_replica_loss = strategy.run(train_fn, args=(dist_data,))
    mean_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=None)
    return mean_loss


@tf.function
def eval_step(dist_data):
    with strategy.scope():
        def eval_fn(input_data):
            token_ids, token_types = input_data['token_ids'], input_data['token_type_ids']
            logits = model((token_ids, token_types))
            loss = computer_loss(token_ids[:, 1:], logits[:, :-1])
            return loss

    per_replica_loss = strategy.run(eval_fn, args=(dist_data,))
    mean_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=None)
    return mean_loss


@tf.function
def test_setp(dist_data):
    with strategy.scope():
        def eval_fn(input_data):
            token_ids, token_types = input_data['token_ids'], input_data['token_type_ids']
            logits = model((token_ids, token_types))
            loss = computer_loss(token_ids[:, 1:], logits[:, :-1])
            return loss

    per_replica_loss = strategy.run(eval_fn, args=(dist_data,))
    mean_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=None)
    return mean_loss


with strategy.scope():
    checkpoint.restore(manager.latest_checkpoint)
    # train
    for epoch in range(EPOCHS):
        train_loss.reset_states()

        for train_data in tqdm(dist_train_dataset):
            trainloss = train_step(train_data)
            train_loss(trainloss)
            manager.save(check_interval=True)
            step = optimizer.iterations
            if step % every_step_save == 0:
                tf.print(f'epoch: {epoch} step: {step.numpy()} loss: {train_loss.result()}')

                # eval
                eval_loss.reset_states()
                tf.print("evaluating....")
                for eval_data in dist_eval_dataset:
                    evaloss = eval_step(eval_data)
                    eval_loss(evaloss)

                tf.print(f"epoch: {epoch} step: {step.numpy()} eval_loss: {eval_loss.result()}")
        if optimizer.iterations > total_step:
            manager.save()
            break

    # test
