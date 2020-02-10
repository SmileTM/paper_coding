# -*- coding: utf-8 -*-
#
# File: modeling.py
# Author: SmileTM
# Site: s-tm.cn
# Github: https://github.com/SmileTM
# Time: 09.02.2020
#
import tensorflow as tf
import utils


class Atttention(tf.keras.layers.Layer):
    def __init__(self,
                 d_hidden=768,
                 n_head=12,
                 dropout_rate=0.0,
                 initializer_range=0.02):
        super(Atttention, self).__init__()
        self.d_hidden = d_hidden
        self.n_head = n_head
        self.dropout_rate = dropout_rate
        self.initializer_range = initializer_range
        self.d_head = d_hidden // n_head

    def build(self, input_shape):
        self.qw = tf.keras.layers.Dense(self.d_hidden, kernel_initializer=get_initializer(self.initializer_range))
        self.kw = tf.keras.layers.Dense(self.d_hidden, kernel_initializer=get_initializer(self.initializer_range))
        self.vw = tf.keras.layers.Dense(self.d_hidden, kernel_initializer=get_initializer(self.initializer_range))
        self.drop = tf.keras.layers.Dropout(rate=self.dropout_rate)

        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.n_head, self.d_head, self.d_hidden),
                                      initializer=get_initializer(self.initializer_range),
                                      trainable=True,
                                      dtype=tf.float32)
        super(Atttention, self).build(input_shape)

    def call(self, inputs):
        q, k, v, atteintion_mask = inputs

        q = self.qw(q)
        k = self.kw(k)
        v = self.vw(v)

        (q, k, v) = self.split_head(q=q, k=k, v=v)

        out = self.attention_procedure(q, k, v, atteintion_mask)
        out = tf.einsum('BNFD->BFND', out)

        return tf.einsum('BFND,NDM->BFM', out, self.kernel)

    def split_head(self, q, k, v):
        batch_size = tf.shape(q)[0]

        q = tf.reshape(q, (batch_size, -1, self.n_head, self.d_head))
        k = tf.reshape(k, (batch_size, -1, self.n_head, self.d_head))
        v = tf.reshape(v, (batch_size, -1, self.n_head, self.d_head))

        q = tf.einsum('BFND->BNFD', q)
        k = tf.einsum('BFND->BNFD', k)
        v = tf.einsum('BFND->BNFD', v)
        return (q, k, v)

    def attention_procedure(self, q, k, v, attention_mask):
        qk = tf.einsum('BNFD,BNfD->BNFf', q, k)
        dk = tf.cast(k.shape[-1], qk.dtype)
        attention_weights = qk / tf.sqrt(dk)
        if attention_mask != None:
            attention_mask = attention_mask[:, None, :, :]
            attention_weights += tf.cast(attention_mask, attention_weights.dtype) * -1e9

        attention_probs = tf.nn.softmax(attention_weights)
        attention_probs = self.drop(attention_probs)

        out = tf.einsum('BNFf,BNfD->BNFD', attention_probs, v)
        return out


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self,
                 n_head=12,
                 d_hidden=768,
                 d_intermediate=3072,
                 hidden_act='gelu',
                 initializer_range=0.02,
                 dropout_rate=0.0):
        super(TransformerBlock, self).__init__()
        self.n_head = n_head
        self.d_hidden = d_hidden
        self.d_intermediate = d_intermediate
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.attention = Atttention(d_hidden=self.d_hidden,
                                    n_head=self.n_head,
                                    dropout_rate=self.dropout_rate)
        self.attention_layer_norm = tf.keras.layers.LayerNormalization(name="self_attention_layer_norm",
                                                                       axis=-1,
                                                                       epsilon=1e-12,
                                                                       dtype=tf.float32)
        self.attention_dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)

        self.output_layer_norm = tf.keras.layers.LayerNormalization(name="output_layer_norm",
                                                                    axis=-1,
                                                                    epsilon=1e-12,
                                                                    dtype=tf.float32)
        self.output_dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)

        self.output_dense1 = tf.keras.layers.Dense(self.d_intermediate,
                                                   activation=utils.get_activation(self.hidden_act),
                                                   )
        self.output_dense2 = tf.keras.layers.Dense(self.d_hidden,
                                                   activation=None,
                                                   )

        super(TransformerBlock, self).build(input_shape)

    def call(self, inputs):
        (input_tensor, attention_mask) = inputs
        attention_output = self.attention((input_tensor, input_tensor, input_tensor, attention_mask))
        attention_output = self.attention_dropout(attention_output)
        attention_output = self.attention_layer_norm(input_tensor + attention_output)

        layer_output = self.output_dense1(attention_output)
        layer_output = self.output_dense2(layer_output)

        layer_output = self.output_dropout(layer_output)
        layer_output = self.output_layer_norm(layer_output + attention_output)

        return layer_output

class Transformer(tf.keras.layers.Layer):
    def __init__(self):
        super(Transformer, self).__init__()
    def build(self, input_shape):
        pass
    def call(self, inputs, **kwargs):
        pass
def get_initializer(initializer_range=0.02):
    """Creates a `tf.initializers.truncated_normal` with the given range.

    Args:
      initializer_range: float, initializer range for stddev.

    Returns:
      TruncatedNormal initializer with stddev = `initializer_range`.
    """
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)






if __name__ == '__main__':
    input_data = tf.constant(tf.random.uniform((32, 512, 768)))
    at = TransformerBlock(d_hidden=768, n_head=12)((input_data, None))
    print(at.shape)

