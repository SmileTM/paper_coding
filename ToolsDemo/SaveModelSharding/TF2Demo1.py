# -*- coding: utf-8 -*-
#
# File: TF2Demo1.py
# Author: SmileTM
# Site: s-tm.cn
# Github: https://github.com/SmileTM
# Time: 06.10.2023
#
import tensorflow as tf


# TF2.x 保存模型方式
class MyModel(tf.keras.Model):
    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        # TF2.x 一定要记住给name赋值，否则会根据整个类图进行增量自动命名
        self.dense1 = tf.keras.layers.Dense(1024, name="dense1")
        self.dense2 = tf.keras.layers.Dense(1024, name="dense2")
        # 设置 save 输入绑定至 serving 函数 (可选)
        # self._set_save_spec(self.serving.input_signature[0])

    def call(self, inputs, training=None, mask=None):
        out = self.dense1(inputs)
        out = self.dense2(out)
        return out

    @tf.function(input_signature=[(
            tf.TensorSpec(shape=(None, 1024), dtype=tf.float32, name="inputs")
    )])
    def serving(self, inputs):
        return {"outputs": self.call(inputs, training=False)}


if __name__ == '__main__':
    saved_path = "saved_tf_demo1"

    data = tf.ones((1024, 1024))
    model = MyModel()
    out = model(data)
    print(out)
    model.save(saved_path, signatures={"serving_default": model.serving})

    '''
    >>> saved_model_cli show --all --dir saved_tf_demo1
    
    signature_def['serving_default']:
      The given SavedModel SignatureDef contains the following input(s):
        inputs['inputs'] tensor_info:
            dtype: DT_FLOAT
            shape: (-1, 1024)
            name: serving_default_inputs:0
      The given SavedModel SignatureDef contains the following output(s):
        outputs['outputs'] tensor_info:
            dtype: DT_FLOAT
            shape: (-1, 1024)
            name: StatefulPartitionedCall:0
      Method name is: tensorflow/serving/predict

    '''
