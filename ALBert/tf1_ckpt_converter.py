# -*- coding: utf-8 -*-
#
# File: tf1_ckpt_converter.py
# Author: SmileTM
# Site: s-tm.cn
# Github: https://github.com/SmileTM
# Time: 02.21.2020
#
import tensorflow as tf
import numpy as np
from absl import flags, app
import models
import re
import os
import modeling

FLAGS = flags.FLAGS
flags.DEFINE_string("albert_config_file", "/Users/lollipop/Downloads/albert_base/albert_config.json",
                    "Bert configuration file to define core bert layers.")

flags.DEFINE_string("new_checkpoint_output_path", "out_new",
                    "Name for the created object-based tf2 checkpoint.")

flags.DEFINE_string(
    "TF1_checkpoint_path", "/Users/lollipop/Downloads/albert_base/model.ckpt-best",
    "Initial checkpoint from a pretrained BERT tf1 model of Google ")

flags.DEFINE_integer("max_seq_length", 512, "Maximum sequence length.")

flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence.")


def re_map_tf1(name):
    # 通过正则来进行模型名字映射
    tensor_name = name
    tensor_name = re.sub("albert/embedding_processor/embedding_pos/embeddings:0",
                         "bert/embeddings/position_embeddings", tensor_name)
    tensor_name = re.sub("albert/embedding_processor/embedding_word_ids/embeddings:0",
                         "bert/embeddings/word_embeddings", tensor_name)
    tensor_name = re.sub("albert/embedding_processor/embedding_type_ids/embeddings:0",
                         "bert/embeddings/token_type_embeddings", tensor_name)
    tensor_name = re.sub("albert/embedding_processor/layer_norm/gamma:0",
                         "bert/embeddings/LayerNorm/gamma", tensor_name)
    tensor_name = re.sub("albert/embedding_processor/layer_norm/beta:0",
                         "bert/embeddings/LayerNorm/beta", tensor_name)
    tensor_name = re.sub("albert/encoder/embedding_hidden_mapping_in/kernel:0",
                         "bert/encoder/embedding_hidden_mapping_in/kernel", tensor_name)
    tensor_name = re.sub("albert/encoder/embedding_hidden_mapping_in/bias:0",
                         "bert/encoder/embedding_hidden_mapping_in/bias", tensor_name)
    tensor_name = re.sub("albert/encoder/transformer/self_attention/query/kernel:0",
                         "bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel", tensor_name)
    tensor_name = re.sub("albert/encoder/transformer/self_attention/query/bias:0",
                         "bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/bias", tensor_name)
    tensor_name = re.sub("albert/encoder/transformer/self_attention/key/kernel:0",
                         "bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel", tensor_name)
    tensor_name = re.sub("albert/encoder/transformer/self_attention/key/bias:0",
                         "bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/bias", tensor_name)
    tensor_name = re.sub("albert/encoder/transformer/self_attention/value/kernel:0",
                         "bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel", tensor_name)
    tensor_name = re.sub("albert/encoder/transformer/self_attention/value/bias:0",
                         "bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/bias", tensor_name)
    tensor_name = re.sub("albert/encoder/transformer/self_attention/self_attention_output/kernel:0",
                         "bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel", tensor_name)
    tensor_name = re.sub("albert/encoder/transformer/self_attention/self_attention_output/bias:0",
                         "bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/bias", tensor_name)
    tensor_name = re.sub("albert/encoder/transformer/self_attention_layer_norm/gamma:0",
                         "bert/encoder/transformer/group_0/inner_group_0/LayerNorm/gamma", tensor_name)
    tensor_name = re.sub("albert/encoder/transformer/self_attention_layer_norm/beta:0",
                         "bert/encoder/transformer/group_0/inner_group_0/LayerNorm/beta", tensor_name)
    tensor_name = re.sub("albert/encoder/transformer/intermediate/kernel:0",
                         "bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel", tensor_name)
    tensor_name = re.sub("albert/encoder/transformer/intermediate/bias:0",
                         "bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/bias", tensor_name)
    tensor_name = re.sub("albert/encoder/transformer/output/kernel:0",
                         "bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel",
                         tensor_name)
    tensor_name = re.sub("albert/encoder/transformer/output/bias:0",
                         "bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/bias",
                         tensor_name)
    tensor_name = re.sub("albert/encoder/transformer/output_layer_norm/gamma:0",
                         "bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/gamma", tensor_name)
    tensor_name = re.sub("albert/encoder/transformer/output_layer_norm/beta:0",
                         "bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/beta", tensor_name)
    tensor_name = re.sub("albert/pooler_transform/kernel:0", "bert/pooler/dense/kernel", tensor_name)
    tensor_name = re.sub("albert/pooler_transform/bias:0", "bert/pooler/dense/bias", tensor_name)

    # pretrain_model
    tensor_name = re.sub("mask_label_loss/output_bias:0", "cls/predictions/output_bias", tensor_name)
    tensor_name = re.sub("mask_label_loss/transform/kernel:0",
                         "cls/predictions/transform/dense/kernel", tensor_name)
    tensor_name = re.sub("mask_label_loss/transform/bias:0", "cls/predictions/transform/dense/bias",
                         tensor_name)
    tensor_name = re.sub("mask_label_loss/layer_norm/gamma:0",
                         "cls/predictions/transform/LayerNorm/gamma", tensor_name)
    tensor_name = re.sub("mask_label_loss/layer_norm/beta:0",
                         "cls/predictions/transform/LayerNorm/beta", tensor_name)
    tensor_name = re.sub("next_sentence_labels/dense/kernel:0",
                         "cls/seq_relationship/output_weights", tensor_name)
    tensor_name = re.sub("next_sentence_labels/dense/bias:0",
                         "cls/seq_relationship/output_bias", tensor_name)
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
        if name == "next_sentence_labels/dense/kernel:0":
            map_tensor = map_tensor.T
        trainable_weight.assign(map_tensor)
        print(f"{map_name,map_tensor.shape}  >>>>  {name,trainable_weight.shape}   转换成功")

    model.save_weights(os.path.join(new_ckpt_save_path, "albert_model.ckpt"))


def main(_):
    assert tf.version.VERSION.startswith('2.')
    config = modeling.AlbertConfig.from_json_file(FLAGS.albert_config_file)
    # 只载入bert模型部分的权重 不载入 预测部分
    model = models.get_base_model(config,max_seq_length=FLAGS.max_seq_length)
    #载入官方预训练 权重（包括预测部分)
    # model = models.getPretrainingModel(config=config,
    #                                    max_predictions_per_seq=FLAGS.max_predictions_per_seq,
    #                                    max_seq_length=FLAGS.max_seq_length)
    conver_model_tf1(model, FLAGS.TF1_checkpoint_path, FLAGS.new_checkpoint_output_path)
    print("TF1模型转换完成")


if __name__ == '__main__':
    flags.mark_flag_as_required("albert_config_file")
    flags.mark_flag_as_required("TF1_checkpoint_path")
    flags.mark_flag_as_required("new_checkpoint_output_path")
    app.run(main)





    # config = modeling.AlbertConfig.from_json_file("/Users/lollipop/Downloads/albert_base/albert_config.json")
    # model = models.getPretrainingModel(config=config,
    #                                    max_predictions_per_seq=512,
    #                                    max_seq_length=512)
    #
    # model.load_weights("/Users/lollipop/Documents/paper_coding/ALBert/out_new/albert_model.ckpt")
    # print(model.trainable_weights)

