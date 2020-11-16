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


# "n_vocab": 50257,
# "n_ctx": 1024,
# "n_embd": 768,
# "n_head": 12,
# "n_layer": 12
class GPT2Config(object):
    def __init__(
            self,
            n_vocab=50257,
            n_positions=1024,
            n_ctx=1024,
            n_embd=768,
            n_head=12,
            n_layer=12,
            n_type=10,
            hidden_act='gelu',
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            resid_pdrop=0.1,
            initializer_range=0.02,
            layer_norm_epsilon=1e-5
    ):
        self.n_vocab = n_vocab
        self.n_positions = n_positions
        self.n_ctx = n_ctx
        self.n_embed = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.n_type = n_type
        self.hidden_act = hidden_act
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.initializer_range = initializer_range
        self.layer_norm_epsilon = layer_norm_epsilon


class Attention(tf.keras.layers.Layer):
    def __init__(self, config: GPT2Config, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.config = config
        self.n_head = config.n_head
        self.hidden_size = config.n_embed

    def build(self, input_shape):
        '''conv1d is just dense layer
        https://github.com/openai/gpt-2/issues/165
        '''
        self.c_attn = tf.keras.layers.Dense(3 * self.hidden_size, name='c_attn',
                                            kernel_initializer=tf.initializers.TruncatedNormal(
                                                stddev=self.config.initializer_range))
        self.c_proj = tf.keras.layers.Dense(self.hidden_size, name='c_proj',
                                            kernel_initializer=tf.initializers.TruncatedNormal(
                                                stddev=self.config.initializer_range))
        self.attn_drop_out = tf.keras.layers.Dropout(self.config.attn_pdrop)
        self.drop_out = tf.keras.layers.Dropout(self.config.resid_pdrop)

    def attention_procedure(self, q, k, v, attention_mask):
        qk = tf.einsum("BFNH,BTNH->BNFT", q, k)
        dk = tf.cast(k.shape[-1], dtype=qk.dtype)
        attention_weights = qk / tf.sqrt(dk)
        if attention_mask is not None:
            attention_weights = attention_weights - (1 - attention_mask) * 1e-6

        attention_probs = tf.nn.softmax(attention_weights, -1)
        attention_probs = self.attn_drop_out(attention_probs)
        attention_out = tf.einsum("BNFT,BTNH->BFNH", attention_probs, v)
        return attention_out

    def call(self, inputs):
        x, attention_mask = inputs
        x = self.c_attn(x)  # [B,L,H*3]
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
    def __init__(self, config: GPT2Config, hidden_size, inner_hidden_size, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.config = config
        self.hidden_size = hidden_size
        self.inner_hidden_size = inner_hidden_size

    def build(self, input_shape):
        self.c_fc = tf.keras.layers.Dense(self.inner_hidden_size, name='c_fc',
                                          kernel_initializer=tf.initializers.TruncatedNormal(
                                              stddev=self.config.initializer_range))
        self.c_proj = tf.keras.layers.Dense(self.hidden_size, name='c_proj',
                                            kernel_initializer=tf.initializers.TruncatedNormal(
                                                stddev=self.config.initializer_range))
        # Todo: replace `tfa.activations.gelu` to `tf.activations.gelu` in tf2.4+
        self.gelu = get_activation(self.config.hidden_act)
        self.mlp_drop_out = tf.keras.layers.Dropout(self.config.resid_pdrop)

    def call(self, inputs):
        h1 = self.gelu(self.c_fc(inputs))
        h2 = self.mlp_drop_out(self.c_proj(h1))
        return h2


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, config: GPT2Config, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.config = config
        self.hidden_size = self.config.n_embed
        self.n_head = self.config.n_head

    def build(self, input_shape):
        '''conv1d is just dense layer
        https://github.com/openai/gpt-2/issues/165
        '''

        self.ln_1 = tf.keras.layers.LayerNormalization(name='ln_1', epsilon=self.config.layer_norm_epsilon)
        self.ln_2 = tf.keras.layers.LayerNormalization(name='ln_2', epsilon=self.config.layer_norm_epsilon)
        self.attenton = Attention(config=self.config, name='attn')
        self.mlp = MLP(config=self.config, hidden_size=self.hidden_size, inner_hidden_size=self.hidden_size * 4,
                       name='mlp')

    def call(self, inputs):
        x, attention_mask = inputs
        a = self.attenton((self.ln_1(x), attention_mask))
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x


class GPT2Model(tf.keras.layers.Layer):
    def __init__(self, config: GPT2Config, **kwargs):
        super(GPT2Model, self).__init__(**kwargs)
        self.config = config
        self.n_positions = config.n_positions

    def build(self, input_shape):
        # position embedding
        self.wpe = tf.keras.layers.Embedding(self.config.n_positions, self.config.n_embed, name='wpe',
                                             embeddings_initializer=tf.random_normal_initializer(
                                                 stddev=0.01))  # pos embed
        # token_id embedding
        self.wte = tf.keras.layers.Embedding(self.config.n_vocab, self.config.n_embed, name='wte',
                                             embeddings_initializer=tf.random_normal_initializer(
                                                 stddev=0.02))
        # token_type_ids embedding
        self.tte = tf.keras.layers.Embedding(self.config.n_type, self.config.n_embed, name='tte',
                                             embeddings_initializer=tf.random_normal_initializer(
                                                 stddev=0.02))
        self.embd_dropout = tf.keras.layers.Dropout(self.config.embd_pdrop)
        self.block_list = [TransformerBlock(self.config, name='h' + str(i)) for i in range(self.config.n_layer)]
        self.ln = tf.keras.layers.LayerNormalization(name='ln_f', epsilon=self.config.layer_norm_epsilon)

    def call(self, inputs, **kwargs):
        token_ids, token_types = inputs
        batch_size, token_ids_length = token_ids.shape

        token_pos = tf.tile(tf.range(token_ids_length)[None, :], [batch_size, 1])

        token_ids_embeds = self.wte(token_ids)
        token_types_embeds = self.tte(token_types)
        token_pos_embeds = self.wpe(token_pos)
        # todo: add past
        nf = token_ids.shape[1]
        nt = token_ids.shape[1]
        attention_mask = self.get_attention_mask(nf, nt, token_ids_embeds.dtype)
        attention_mask = tf.reshape(attention_mask, [1, 1, nf, nt])
        hidden_states = token_ids_embeds + token_types_embeds + token_pos_embeds
        hidden_states = self.embd_dropout(hidden_states)
        for tfblock in self.block_list:
            hidden_states = tfblock((hidden_states, attention_mask))

        hidden_states = self.ln(hidden_states)
        logits = tf.matmul(hidden_states, self.wte.embeddings, transpose_b=True)

        return logits

    def get_attention_mask(self, nf, nt, dtype):
        i = tf.range(nf)[:, None]
        j = tf.range(nt)
        m = i >= j
        return tf.cast(m, dtype)


def get_activation(name_str):
    # Todo: change tfa to tf until tf2.4
    # tf.keras.activations.relu
    actionsDict = {'gelu': tfa.activations.gelu,
                   'relu': tf.keras.activations.relu,
                   'tanh': tf.keras.activations.tanh}
    return actionsDict[name_str]


if __name__ == '__main__':
    config = GPT2Config()
    model = GPT2Model(config, name='model')
    token_ids = tf.ones((2, 10))
    token_types = tf.ones((2, 10))
    print(model((token_ids, token_types)))
    for i in model.trainable_variables:
        print(i.name, i.numpy().shape)
