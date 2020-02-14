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
import copy


class BertConfig(object):
    def __init__(self,
                 vocab_size=20000,
                 n_head=12,
                 d_hidden=768,
                 num_hidden_layers=12,
                 d_intermediate=3072,
                 max_position_embedding=512,
                 type_vocab_size=16,
                 hidden_act='gelu',
                 initializer_range=0.02,
                 attention_dropout_rate=0.0,
                 hidden_dropout_rate=0.0):
        self.vocab_size = vocab_size
        self.n_head = n_head
        self.d_hidden = d_hidden
        self.num_hidden_layers = num_hidden_layers
        self.d_intermediate = d_intermediate
        self.max_position_embedding = max_position_embedding
        self.type_vocab_size = type_vocab_size
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.attention_dropout_rate = attention_dropout_rate
        self.hidden_dropout_rate = hidden_dropout_rate

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output


class BertModel(tf.keras.layers.Layer):
    def __init__(self, bertconfig: BertConfig, **kwargs):
        super(BertModel, self).__init__(**kwargs)
        self.config = bertconfig

    def build(self, input_shape):
        self.encoder = Transformer(num_hidden_layers=self.config.num_hidden_layers,
                                   d_hidden=self.config.d_hidden,
                                   n_head=self.config.n_head,
                                   d_intermediate=self.config.d_intermediate,
                                   hidden_act=self.config.hidden_act,
                                   attention_dropout_rate=self.config.attention_dropout_rate,
                                   hidden_dropout_rate=self.config.hidden_dropout_rate,
                                   initializer_range=self.config.initializer_range,
                                   name='encoder'
                                   )
        self.embedding_processor = EmbeddingProcessor(vocab_szie=self.config.vocab_size,
                                                      d_hidden=self.config.d_hidden,
                                                      max_position_embedding=self.config.max_position_embedding,
                                                      type_vocab_size=self.config.type_vocab_size,
                                                      hidden_dropout_rate=self.config.hidden_dropout_rate,
                                                      initializer_range=self.config.initializer_range)
        self.pooler_transform = tf.keras.layers.Dense(
            units=self.config.d_hidden,
            activation="tanh",
            kernel_initializer=get_initializer(self.config.initializer_range),
            name="pooler_transform")

        super(BertModel, self).build(input_shape)

    def call(self, inputs, mode="bert"):
        (input_word_ids, input_mask, input_type_ids) = inputs
        input_tensor = self.embedding_processor((input_word_ids, input_type_ids))
        attention_mask = create_attention_mask_from_input_mask(input_mask)

        if mode == 'encoder':
            return self.encoder((input_tensor, attention_mask), return_all_layers=True)

        sequence_output = self.encoder((input_tensor, attention_mask))
        first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)  # [batch_size ,d_hidden]
        pooled_output = self.pooler_transform(first_token_tensor)

        return (pooled_output, sequence_output)

    def get_embedding_table(self):
        return self.embedding_processor.embedding_word_ids.embeddings

    def get_config(self):
        config = super(BertModel, self).get_config()
        config.update({"config": self.config.to_dict()})
        return config


class EmbeddingProcessor(tf.keras.layers.Layer):
    def __init__(self,
                 vocab_szie,
                 d_hidden=768,
                 max_position_embedding=512,
                 type_vocab_size=16,
                 hidden_dropout_rate=0.0,
                 initializer_range=0.02,
                 **kwargs):
        super(EmbeddingProcessor, self).__init__(**kwargs)
        self.vocab_size = vocab_szie
        self.d_hidden = d_hidden
        self.max_position_embedding = max_position_embedding
        self.type_vocab_size = type_vocab_size
        self.hidden_dropout_rate = hidden_dropout_rate
        self.initializer_range = initializer_range

    def build(self, input_shape):
        self.embedding_word_ids = tf.keras.layers.Embedding(input_dim=self.vocab_size,
                                                            output_dim=self.d_hidden,
                                                            embeddings_initializer=get_initializer(
                                                                self.initializer_range),
                                                            name="embedding_word_ids",
                                                            dtype=tf.float32
                                                            )
        self.embedding_type_ids = tf.keras.layers.Embedding(input_dim=self.type_vocab_size,
                                                            output_dim=self.d_hidden,
                                                            embeddings_initializer=get_initializer(
                                                                self.initializer_range),
                                                            name="embedding_type_ids",
                                                            dtype=tf.float32)
        self.embedding_pos = self.add_weight(name='embedding_pos',
                                             shape=(self.max_position_embedding, self.d_hidden),
                                             initializer=get_initializer(self.initializer_range),
                                             dtype=tf.float32)

        self.output_layer_norm = tf.keras.layers.LayerNormalization(
            name="layer_norm", axis=-1, epsilon=1e-12, dtype=tf.float32)

        self.output_dropout = tf.keras.layers.Dropout(
            rate=self.hidden_dropout_rate, dtype=tf.float32)
        super(EmbeddingProcessor, self).build(input_shape)

    def call(self, inputs):
        input_word_ids, input_type_ids = inputs
        token_word_embeddings = self.embedding_word_ids(input_word_ids)  # [batch_size, seq_length, hidden_size]
        token_type_embeddings = self.embedding_type_ids(input_type_ids)  # [batch_size, seq_length, hidden_size]
        token_pos_embeddings = tf.expand_dims(self.embedding_pos, axis=0)  # [1, seq_length, hidden_size]

        output = token_word_embeddings + token_type_embeddings + token_pos_embeddings
        output = self.output_layer_norm(output)
        output = self.output_dropout(output)
        return output


class Atttention(tf.keras.layers.Layer):
    def __init__(self,
                 d_hidden=768,
                 n_head=12,
                 dropout_rate=0.0,
                 initializer_range=0.02,
                 **kwargs):
        super(Atttention, self).__init__(**kwargs)
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

        if attention_mask is not None:
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
                 hidden_dropout_rate=0.0,
                 attention_dropout_rate=0.0,
                 **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.n_head = n_head
        self.d_hidden = d_hidden
        self.d_intermediate = d_intermediate
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.hidden_dropout_rate = hidden_dropout_rate
        self.attention_dropout_rate = attention_dropout_rate

    def build(self, input_shape):
        self.attention = Atttention(d_hidden=self.d_hidden,
                                    n_head=self.n_head,
                                    dropout_rate=self.attention_dropout_rate)
        self.attention_layer_norm = tf.keras.layers.LayerNormalization(name="self_attention_layer_norm",
                                                                       axis=-1,
                                                                       epsilon=1e-12,
                                                                       dtype=tf.float32)
        self.attention_dropout = tf.keras.layers.Dropout(rate=self.hidden_dropout_rate)

        self.output_layer_norm = tf.keras.layers.LayerNormalization(name="output_layer_norm",
                                                                    axis=-1,
                                                                    epsilon=1e-12,
                                                                    dtype=tf.float32)
        self.output_dropout = tf.keras.layers.Dropout(rate=self.hidden_dropout_rate)

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
    def __init__(self,
                 n_head=12,
                 d_hidden=768,
                 num_hidden_layers=12,
                 d_intermediate=3072,
                 hidden_act='gelu',
                 initializer_range=0.02,
                 attention_dropout_rate=0.0,
                 hidden_dropout_rate=0.0,
                 **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.n_head = n_head
        self.d_hidden = d_hidden
        self.num_hidden_layers = num_hidden_layers
        self.d_intermerdiate = d_intermediate
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.hidden_dropout_rate = hidden_dropout_rate
        self.attention_dropout_rate = attention_dropout_rate

    def build(self, input_shape):
        self.layers = []

        for i in range(self.num_hidden_layers):
            self.layers.append(TransformerBlock(n_head=self.n_head,
                                                d_hidden=self.d_hidden,
                                                d_intermediate=self.d_intermerdiate,
                                                hidden_act=self.hidden_act,
                                                initializer_range=self.initializer_range,
                                                hidden_dropout_rate=self.hidden_dropout_rate,
                                                attention_dropout_rate=self.attention_dropout_rate,
                                                name=("layer_%d" % i)))

        super(Transformer, self).build(input_shape)

    def call(self, inputs, return_all_layers=False):
        input_tensro, attention_mask = inputs
        output_tensor = input_tensro
        all_layer_outputs = []
        for layer in self.layers:
            output_tensor = layer((output_tensor, attention_mask))
            all_layer_outputs.append(output_tensor)
        if return_all_layers:
            return all_layer_outputs
        return all_layer_outputs[-1]


def get_initializer(initializer_range=0.02):
    """Creates a `tf.initializers.truncated_normal` with the given range.

    Args:
      initializer_range: float, initializer range for stddev.

    Returns:
      TruncatedNormal initializer with stddev = `initializer_range`.
    """
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


def create_attention_mask_from_input_mask(mask):
    '''

    :param mask: shepe = [batch_size, seq_length]
    :return: attentino_mask = shape = [batch_size, seq_length, seq_length]
    '''
    from_mask = mask[:, :, None]
    to_mask = mask[:, None, :]
    attention_mask = from_mask * to_mask
    return tf.cast(attention_mask, tf.float32)


if __name__ == '__main__':
    input_word_ids = tf.keras.layers.Input(shape=(512,))
    input_mask = tf.keras.layers.Input(shape=(512,))
    input_type_ids = tf.keras.layers.Input(shape=(512,))

    config = BertConfig(max_position_embedding=512)
    bertModel = BertModel(config)
    output = bertModel((input_word_ids, input_mask, input_type_ids))

    model = tf.keras.Model(inputs=(input_type_ids, input_mask, input_word_ids), outputs=output)
    #
    # print(model.trainable_weights)
    # model.summary()
    model.load_weights('./out/allmodel-ckpt')
    # print(model.trainable_weights)
    model.trainable_weights[-1].numpy= tf.random.uniform(shape=(768,),dtype=tf.float32)
    model.layers[-1].trainable_weights[-1].assign(tf.ones(shape=(768,),dtype=tf.float32))
    print(model.layers[-1].trainable_weights[-1])

    print(model.trainable_weights)

