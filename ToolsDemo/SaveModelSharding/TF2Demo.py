# -*- coding: utf-8 -*-
#
# File: TF2Demo.py
# Author: SmileTM
# Site: s-tm.cn
# Github: https://github.com/SmileTM
# Time: 06.10.2023
#
from dataclasses import dataclass
import tensorflow as tf
from transformers import PretrainedConfig, TFPreTrainedModel
from transformers.utils import ModelOutput
import os


# TF2.x 使用HuggingFace transformers保存模型方式

class ModelConfig(PretrainedConfig):
    model_type = "tfmodel"

    def __init__(self, size=1024, **kwargs):
        super().__init__(**kwargs)
        self.size = size


config = ModelConfig(size=1024)


@dataclass
class ModelOut(ModelOutput):
    logits: tf.Tensor = None


class Model(TFPreTrainedModel):

    @property
    def dummy_inputs(self):
        dummy = tf.constant(tf.ones((self.config.size, self.config.size), dtype=tf.float32))
        return dummy

    def __init__(self, config: ModelConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config

        # TF2.x 一定要记住给name赋值，否则会根据整个类图进行增量自动命名
        self.dense1 = tf.keras.layers.Dense(self.config.size, name="dense1")
        self.dense2 = tf.keras.layers.Dense(self.config.size, name="dense2")

    def call(self, inputs, training=None, mask=None):
        out = self.dense1(inputs)
        out = self.dense2(out)
        return out

    @tf.function(
        input_signature=[
            (tf.TensorSpec(shape=(None, config.size), dtype=tf.float32, name="inputs"))
        ]
    )
    def serving(self, inputs):
        output = self.call(inputs)
        return self.serving_output(output)

    def serving_output(self, output):
        return ModelOut(logits=output)


if __name__ == '__main__':
    saved_path = "saved_tf"
    os.makedirs(saved_path, exist_ok=True)

    config = ModelConfig(size=1024)
    data = tf.ones((1024, 1024))
    model = Model(config=config)
    out = model(data)
    print(f"model output:", out)
    model.save_pretrained(saved_path, max_shard_size="6MB")
    # 打印model参数
    print(model.trainable_variables)

    model1 = Model(config=config).from_pretrained(saved_path, config=config)
    out1 = model1(data)
    print(f"model1 output:", out)
    # 打印model1参数
    print(model1.trainable_variables)

    model.save(saved_path + "_raw")

    '''
    >>> tree saved_tf

    saved_tf
    ├── config.json
    ├── tf_model-00001-of-00002.h5
    ├── tf_model-00002-of-00002.h5
    └── tf_model.h5.index.json

    '''
