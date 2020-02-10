# -*- coding: utf-8 -*-
#
# File: utils.py
# Author: SmileTM
# Site: s-tm.cn
# Github: https://github.com/SmileTM
# Time: 10.02.2020
#
import math
import tensorflow as tf

def gelu(x):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf

def get_activation(act_name:str):
    act_name = act_name.lower()
    name_to_fn = {
        'gelu':gelu
    }
    if act_name not in name_to_fn:
        return tf.keras.activations.get(act_name)
    return name_to_fn[act_name]
