# -*- coding: utf-8 -*-
#
# File: tf1_ckpt_converter.py
# Author: SmileTM
# Site: s-tm.cn
# Github: https://github.com/SmileTM
# Time: 02.20.2020
#


import tensorflow as tf
import numpy as np
from absl import flags, app
import models
import re
import os
import modeling

FLAGS = flags.FLAGS
flags.DEFINE_string("bert_config_file", "/Users/lollipop/Downloads/bert/chinese_L-12_H-768_A-12/bert_config.json",
                    "Bert configuration file to define core bert layers.")

flags.DEFINE_string("new_checkpoint_output_path", "out_new",
                    "Name for the created object-based tf2 checkpoint.")

flags.DEFINE_string(
    "TF1_checkpoint_path", "/Users/lollipop/Downloads/bert/chinese_L-12_H-768_A-12/bert_model.ckpt",
    "Initial checkpoint from a pretrained BERT tf1 model of Google ")

flags.DEFINE_integer("max_seq_length", 512, "Maximum sequence length.")

flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence.")
def re_map_tf1(name):
    # 通过正则来进行模型名字映射
    tensor_name = name
    tensor_name = re.sub(r"bert/encoder/layer_(\d+)/self_attention/query/kernel:0",
                         r"bert/encoder/layer_\1/attention/self/query/kernel", tensor_name)
    tensor_name = re.sub(r"bert/encoder/layer_(\d+)/self_attention/query/bias:0",
                         r"bert/encoder/layer_\1/attention/self/query/bias", tensor_name)
    tensor_name = re.sub(r"bert/encoder/layer_(\d+)/self_attention/key/kernel:0",
                         r"bert/encoder/layer_\1/attention/self/key/kernel", tensor_name)
    tensor_name = re.sub(r"bert/encoder/layer_(\d+)/self_attention/key/bias:0",
                         r"bert/encoder/layer_\1/attention/self/key/bias", tensor_name)
    tensor_name = re.sub(r"bert/encoder/layer_(\d+)/self_attention/value/kernel:0",
                         r"bert/encoder/layer_\1/attention/self/value/kernel", tensor_name)
    tensor_name = re.sub(r"bert/encoder/layer_(\d+)/self_attention/value/bias:0",
                         r"bert/encoder/layer_\1/attention/self/value/bias", tensor_name)
    tensor_name = re.sub(r"bert/encoder/layer_(\d+)/self_attention/self_attention_output/kernel:0",
                         r"bert/encoder/layer_\1/attention/output/dense/kernel", tensor_name)
    tensor_name = re.sub(r"bert/encoder/layer_(\d+)/self_attention/self_attention_output/bias:0",
                         r"bert/encoder/layer_\1/attention/output/dense/bias", tensor_name)
    tensor_name = re.sub(r"bert/encoder/layer_(\d+)/self_attention_layer_norm/gamma:0",
                         r"bert/encoder/layer_\1/attention/output/LayerNorm/gamma", tensor_name)
    tensor_name = re.sub(r"bert/encoder/layer_(\d+)/self_attention_layer_norm/beta:0",
                         r"bert/encoder/layer_\1/attention/output/LayerNorm/beta", tensor_name)
    tensor_name = re.sub(r"bert/encoder/layer_(\d+)/intermediate/kernel:0",
                         r"bert/encoder/layer_\1/intermediate/dense/kernel", tensor_name)
    tensor_name = re.sub(r"bert/encoder/layer_(\d+)/intermediate/bias:0",
                         r"bert/encoder/layer_\1/intermediate/dense/bias", tensor_name)
    tensor_name = re.sub(r"bert/encoder/layer_(\d+)/output/kernel:0", r"bert/encoder/layer_\1/output/dense/kernel",
                         tensor_name)
    tensor_name = re.sub(r"bert/encoder/layer_(\d+)/output/bias:0", r"bert/encoder/layer_\1/output/dense/bias",
                         tensor_name)
    tensor_name = re.sub(r"bert/encoder/layer_(\d+)/output_layer_norm/gamma:0",
                         r"bert/encoder/layer_\1/output/LayerNorm/gamma", tensor_name)
    tensor_name = re.sub(r"bert/encoder/layer_(\d+)/output_layer_norm/beta:0",
                         r"bert/encoder/layer_\1/output/LayerNorm/beta", tensor_name)

    tensor_name = re.sub(r"bert/embedding_processor/embedding_pos/embeddings:0", r"bert/embeddings/position_embeddings",
                         tensor_name)
    tensor_name = re.sub(r"bert/embedding_processor/embedding_word_ids/embeddings:0",
                         r"bert/embeddings/word_embeddings", tensor_name)
    tensor_name = re.sub(r"bert/embedding_processor/embedding_type_ids/embeddings:0",
                         r"bert/embeddings/token_type_embeddings", tensor_name)
    tensor_name = re.sub(r"bert/embedding_processor/layer_norm/gamma:0", r"bert/embeddings/LayerNorm/gamma",
                         tensor_name)
    tensor_name = re.sub(r"bert/embedding_processor/layer_norm/beta:0", r"bert/embeddings/LayerNorm/beta", tensor_name)
    tensor_name = re.sub(r"bert/pooler_transform/kernel:0", r"bert/pooler/dense/kernel", tensor_name)
    tensor_name = re.sub(r"bert/pooler_transform/bias:0", r"bert/pooler/dense/bias", tensor_name)
    tensor_name = re.sub(r"pretraining_mask_label_loss_layer/output_bias:0", r"cls/predictions/output_bias",
                         tensor_name)
    tensor_name = re.sub(r"pretraining_mask_label_loss_layer/dense/kernel:0", r"cls/predictions/transform/dense/kernel",
                         tensor_name)
    tensor_name = re.sub(r"pretraining_mask_label_loss_layer/dense/bias:0", r"cls/predictions/transform/dense/bias",
                         tensor_name)
    tensor_name = re.sub(r"pretraining_mask_label_loss_layer/layer_norm/gamma:0",
                         r"cls/predictions/transform/LayerNorm/gamma", tensor_name)
    tensor_name = re.sub(r"pretraining_mask_label_loss_layer/layer_norm/beta:0",
                         r"cls/predictions/transform/LayerNorm/beta", tensor_name)
    tensor_name = re.sub(r"next_sentence_loss/dense_1/kernel:0", r"cls/seq_relationship/output_weights", tensor_name)
    tensor_name = re.sub(r"next_sentence_loss/dense_1/bias:0", r"cls/seq_relationship/output_bias", tensor_name)

    return tensor_name


def name_map_tf1(name):
    map_name = re_map_tf1(name)
    return map_name


def conver_model_tf1(model, tf1_ckpt_path, new_ckpt_save_path):
    """Converts a V1 checkpoint of Google into an V2 checkpoint."""
    ckpt_tf1 = tf.train.load_checkpoint(tf1_ckpt_path)
    for trainable_weight in model.trainable_weights:
        name = trainable_weight.name
        # print('name',name)
        map_name = name_map_tf1(name)
        # print('map_name',map_name)
        map_tensor = ckpt_tf1.get_tensor(map_name)
        if name == "next_sentence_loss/dense_1/kernel:0":
            map_tensor = map_tensor.T
        trainable_weight.assign(map_tensor)
        print(f"{name}  >>>>  {map_name}   转换成功")

    model.save_weights(os.path.join(new_ckpt_save_path, "bert_model.ckpt"))


def main(_):
    assert tf.version.VERSION.startswith('2.')
    config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    model = models.getPretrainingModel(config=config,
                                       max_predictions_per_seq=FLAGS.max_predictions_per_seq,
                                       max_seq_length=FLAGS.max_seq_length)
    conver_model_tf1(model, FLAGS.TF1_checkpoint_path, FLAGS.new_checkpoint_output_path)
    print("TF1模型转换完成")


if __name__ == '__main__':
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("TF1_checkpoint_path")
    flags.mark_flag_as_required("new_checkpoint_output_path")
    app.run(main)
