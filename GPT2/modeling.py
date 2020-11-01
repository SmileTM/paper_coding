# -*- coding: utf-8 -*-
#
# File: modeling.py
# Author: SmileTM
# Site: s-tm.cn
# Github: https://github.com/SmileTM
# Time: 10.31.2020
#
import tensorflow as tf
import tensorflow_addons as tfa


class Attention(tf.keras.layers.Layer):
    def __init__(self, n_head, hidden_size, **kwargs):
        self.n_head = n_head
        self.hidden_size = hidden_size

    def build(self, input_shape):
        '''conv1d is just dense layer
        https://github.com/openai/gpt-2/issues/165
        '''
        # Todo: add init
        self.c_attn = tf.keras.layers.Dense(3 * self.hidden_size, name='c_attn')
        self.c_proj = tf.keras.layers.Dense(self.hidden_size, name='c_proj')
        # Todo: config.drop_rate
        self.attn_drop_out = tf.keras.layers.Dropout(0.01)
        self.drop_out = tf.keras.layers.Dropout(0.01)

    def attention_procedure(self, q, k, v, attention_mask):
        qk = tf.einsum("BFNH,BTNH->BNFT", q, k)
        dk = tf.cast(k.shape[-1], dtype=qk.dtype)
        qk = qk / tf.sqrt(dk)
        attention_weights = qk + attention_mask
        if attention_mask is not None:
            attention_weights = attention_weights + attention_mask

        # Todo: add_lm_mask
        attention_probs = tf.nn.softmax(attention_weights, -1)
        attention_probs = self.attn_drop_out(attention_probs)
        attention_out = tf.einsum("BNFT,BTNH->BFNH", attention_probs, v)
        return attention_out

    def call(self, inputs):
        x, attention_mask = inputs
        x = self.c_attn(inputs)  # [B,L,H*3]
        *start, _ = x.shape
        q, k, v = tf.split(x, 3, axis=-1)  # [B,L,H]
        new_shape = start + [self.n_head, -1]
        q = tf.reshape(q, new_shape)  # [B,L,N,H]
        k = tf.reshape(k, new_shape)
        v = tf.reshape(v, new_shape)
        attention_out = self.attention_procedure(q, k, v, attention_mask)
        attention_out = tf.reshape(attention_out, start + [-1])
        c_proj_out = self.c_proj(attention_out)
        out = self.drop_out(c_proj_out)
        return out


class MLP(tf.keras.layers.Layer):
    def __init__(self, hidden_size, inner_hidden_size, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.inner_hidden_size = inner_hidden_size

    def build(self, input_shape):
        # Todo: add init
        self.c_fc = tf.keras.layers.Dense(self.inner_hidden_size, name='c_fc')
        self.c_proj = tf.keras.layers.Dense(self.hidden_size, name='c_proj')
        # Todo: replace `tfa.activations.gelu` to `tf.activations.gelu` in tf2.4+
        self.glue = tfa.activations.gelu()
        # Todo: add config_drop_out_rate
        self.mlp_drop_out = tf.keras.layers.Dropout()

    def call(self, inputs):
        h1 = self.gelu(self.c_fc(inputs))
        h2 = self.mlp_drop_out(self.c_proj(h1))
        return h2


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, n_head, hidden_size, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.n_head = n_head

    def build(self, input_shape):
        '''conv1d is just dense layer
        https://github.com/openai/gpt-2/issues/165
        '''
        # Todo: add config epsilon
        self.ln_1 = tf.keras.layers.LayerNormalization(name='ln_1')
        self.ln_2 = tf.keras.layers.LayerNormalization(name='ln_2')
        self.attenton = Attention(n_head=self.n_head, hidden_size=self.hidden_size, name='attn')
        self.mlp = MLP(hidden_size=self.hidden_size, inner_hidden_size=self.hidden_size * 4, name='mlp')

    def call(self, inputs):
        x, attention_mask = inputs
        a = self.attenton((self.ln_1(x), attention_mask))
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x


class GPT2Model(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(GPT2Model, self).__init__(**kwargs)
        self.config = config

    def build(self, input_shape):
        # Todo: replace to config
        self.wpe = tf.keras.layers.Embedding(1024, 768, name='wpe')
        self.wte = tf.keras.layers.Embedding(50257, 768, name='wte')
        self.block_list = [TransformerBlock(n_head, hidden_size, name='h' + str(i)) for i in range(self.config.n_block)]

    def call(self, inputs, **kwargs):
        token_ids, token_types, token_mask = inputs
        batch_size, token_ids_length = token_ids.shape

        token_pos = tf.tile(tf.range(token_ids_length)[None, :], [batch_size, 1])

        token_ids_embeds = self.wte(token_ids)
        token_types_embeds = self.wte(token_types)
        token_pos_embeds = self.wte(token_pos)

        hidden_states = token_ids_embeds + token_types_embeds + token_pos_embeds
        for tfblock in self.block_list:
            hidden_states = tfblock((hidden_states, attention_mask))

    return hidden_states
