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
import json
import six


class AlbertConfig(object):
    def __init__(self,
                 vocab_size=21128,  # 字典大小
                 embedding_size=128,
                 hidden_size=768,  # 隐藏层维度
                 num_hidden_layers=12,  # Transformer层数
                 num_attention_heads=12,  # head个数
                 intermediate_size=3072,  # FFN中Dense的维度
                 hidden_act='gelu',  # 激活函数
                 hidden_dropout_prob=0.0,  # attention外部的droprate
                 attention_probs_dropout_prob=0.0,  # attnetion中droprate
                 max_position_embeddings=512,  # 最大输入的长度
                 type_vocab_size=16,  # vocab种类
                 initializer_range=0.02):  # 初始化率

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = AlbertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with tf.io.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class AlbertModel(tf.keras.layers.Layer):
    def __init__(self, bertconfig: AlbertConfig, **kwargs):
        super(AlbertModel, self).__init__(**kwargs)
        self.config = bertconfig

    def build(self, input_shape):
        self.embedding_processor = EmbeddingProcessor(vocab_szie=self.config.vocab_size,
                                                      width=self.config.embedding_size,
                                                      max_position_embeddings=self.config.max_position_embeddings,
                                                      type_vocab_size=self.config.type_vocab_size,
                                                      hidden_dropout_prob=self.config.hidden_dropout_prob,
                                                      initializer_range=self.config.initializer_range)

        self.encoder = Transformer(num_hidden_layers=self.config.num_hidden_layers,
                                   hidden_size=self.config.hidden_size,
                                   num_attention_heads=self.config.num_attention_heads,
                                   intermediate_size=self.config.intermediate_size,
                                   hidden_act=self.config.hidden_act,
                                   attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
                                   hidden_dropout_prob=self.config.hidden_dropout_prob,
                                   initializer_range=self.config.initializer_range,
                                   name='encoder'
                                   )

        self.pooler_transform = tf.keras.layers.Dense(units=self.config.hidden_size,
                                                      activation="tanh",
                                                      kernel_initializer=get_initializer(self.config.initializer_range),
                                                      name="pooler_transform")

        super(AlbertModel, self).build(input_shape)

    def call(self, inputs, mode="bert"):
        (input_ids, input_mask, segment_ids) = inputs
        input_tensor = self.embedding_processor((input_ids, segment_ids))
        attention_mask = create_attention_mask_from_input_mask(input_mask)
        if mode == 'encoder':
            return self.encoder((input_tensor, attention_mask), return_all_layers=True)

        sequence_output = self.encoder((input_tensor, attention_mask))
        first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)  # [batch_size ,hidden_size]
        pooled_output = self.pooler_transform(first_token_tensor)

        return (pooled_output, sequence_output)

    def get_embedding_table(self):
        return self.embedding_processor.embedding_word_ids.embeddings

    def get_config(self):
        config = super(AlbertModel, self).get_config()
        config.update({"config": self.config.to_dict()})
        return config


class EmbeddingProcessor(tf.keras.layers.Layer):
    def __init__(self,
                 vocab_szie,
                 width=768,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 hidden_dropout_prob=0.0,
                 initializer_range=0.02,
                 **kwargs):
        super(EmbeddingProcessor, self).__init__(**kwargs)
        self.vocab_size = vocab_szie
        self.width = width
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.initializer_range = initializer_range

    def build(self, input_shape):
        self.embedding_word_ids = tf.keras.layers.Embedding(input_dim=self.vocab_size,
                                                            output_dim=self.width,
                                                            embeddings_initializer=get_initializer(
                                                                self.initializer_range),
                                                            name="embedding_word_ids",
                                                            dtype=tf.float32
                                                            )
        self.embedding_type_ids = tf.keras.layers.Embedding(input_dim=self.type_vocab_size,
                                                            output_dim=self.width,
                                                            embeddings_initializer=get_initializer(
                                                                self.initializer_range),
                                                            name="embedding_type_ids",
                                                            dtype=tf.float32)
        self.embedding_pos = self.add_weight(name='embedding_pos/embeddings',
                                             shape=(self.max_position_embeddings, self.width),
                                             initializer=get_initializer(self.initializer_range),
                                             dtype=tf.float32)

        self.output_layer_norm = tf.keras.layers.LayerNormalization(
            name="layer_norm", axis=-1, epsilon=1e-12, dtype=tf.float32)

        self.output_dropout = tf.keras.layers.Dropout(
            rate=self.hidden_dropout_prob, dtype=tf.float32)
        super(EmbeddingProcessor, self).build(input_shape)

    def call(self, inputs):
        input_ids, segment_ids = inputs
        token_word_embeddings = self.embedding_word_ids(input_ids)  # [batch_size, seq_length, hidden_size]
        token_type_embeddings = self.embedding_type_ids(segment_ids)  # [batch_size, seq_length, hidden_size]
        token_pos_embeddings = tf.expand_dims(self.embedding_pos, axis=0)  # [1, seq_length, hidden_size]

        output = token_word_embeddings + token_type_embeddings + token_pos_embeddings
        output = self.output_layer_norm(output)
        output = self.output_dropout(output)
        return output


class Atttention(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_size=768,
                 num_attention_heads=12,
                 dropout_rate=0.0,
                 initializer_range=0.02,
                 **kwargs):
        super(Atttention, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.dropout_rate = dropout_rate
        self.initializer_range = initializer_range
        self.d_head = hidden_size // num_attention_heads  # 每个头的维度

    def build(self, input_shape):
        self.qw = tf.keras.layers.Dense(self.hidden_size,
                                        kernel_initializer=get_initializer(self.initializer_range),
                                        name='query')
        self.kw = tf.keras.layers.Dense(self.hidden_size,
                                        kernel_initializer=get_initializer(self.initializer_range),
                                        name='key')
        self.vw = tf.keras.layers.Dense(self.hidden_size,
                                        kernel_initializer=get_initializer(self.initializer_range),
                                        name='value')
        self.drop = tf.keras.layers.Dropout(rate=self.dropout_rate)

        self.outdense = tf.keras.layers.Dense(self.hidden_size,
                                              kernel_initializer=get_initializer(self.initializer_range),
                                              name='self_attention_output')

        super(Atttention, self).build(input_shape)

    def call(self, inputs):
        q, k, v, atteintion_mask = inputs

        q = self.qw(q)
        k = self.kw(k)
        v = self.vw(v)

        (q, k, v) = self.split_head(q=q, k=k, v=v)

        out = self.attention_procedure(q, k, v, atteintion_mask)
        out = tf.einsum('BNFD->BFND', out)
        out = tf.reshape(tensor=out, shape=[-1, out.shape[1], out.shape[2] * out.shape[3]])

        out = self.outdense(out)
        return out

    def split_head(self, q, k, v):
        batch_size = tf.shape(q)[0]

        q = tf.reshape(q, (batch_size, -1, self.num_attention_heads, self.d_head))
        k = tf.reshape(k, (batch_size, -1, self.num_attention_heads, self.d_head))
        v = tf.reshape(v, (batch_size, -1, self.num_attention_heads, self.d_head))

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
                 num_attention_heads=12,
                 hidden_size=768,
                 intermediate_size=3072,
                 hidden_act='gelu',
                 initializer_range=0.02,
                 hidden_dropout_prob=0.0,
                 attention_probs_dropout_prob=0.0,
                 **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob

    def build(self, input_shape):
        self.attention = Atttention(hidden_size=self.hidden_size,
                                    num_attention_heads=self.num_attention_heads,
                                    dropout_rate=self.attention_probs_dropout_prob,
                                    name="self_attention")
        self.attention_layer_norm = tf.keras.layers.LayerNormalization(name="self_attention_layer_norm",
                                                                       axis=-1,
                                                                       epsilon=1e-12,
                                                                       dtype=tf.float32)
        self.attention_dropout = tf.keras.layers.Dropout(rate=self.hidden_dropout_prob)

        self.output_dropout = tf.keras.layers.Dropout(rate=self.hidden_dropout_prob)

        self.output_dense1 = tf.keras.layers.Dense(self.intermediate_size,
                                                   activation=utils.get_activation(self.hidden_act),
                                                   name='intermediate'
                                                   )
        self.output_dense2 = tf.keras.layers.Dense(self.hidden_size,
                                                   activation=None,
                                                   name='output'
                                                   )

        self.output_layer_norm = tf.keras.layers.LayerNormalization(name="output_layer_norm",
                                                                    axis=-1,
                                                                    epsilon=1e-12,
                                                                    dtype=tf.float32)

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
                 num_attention_heads=12,
                 hidden_size=768,
                 num_hidden_layers=12,
                 intermediate_size=3072,
                 hidden_act='gelu',
                 initializer_range=0.02,
                 attention_probs_dropout_prob=0.0,
                 hidden_dropout_prob=0.0,
                 **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob

    def build(self, input_shape):
        self.embedding_hidden_map = tf.keras.layers.Dense(self.hidden_size,
                                                          kernel_initializer=get_initializer(self.initializer_range),
                                                          name="embedding_hidden_mapping_in")

        self.layers = [TransformerBlock(num_attention_heads=self.num_attention_heads,
                                        hidden_size=self.hidden_size,
                                        intermediate_size=self.intermediate_size,
                                        hidden_act=self.hidden_act,
                                        initializer_range=self.initializer_range,
                                        hidden_dropout_prob=self.hidden_dropout_prob,
                                        attention_probs_dropout_prob=self.attention_probs_dropout_prob,
                                        name=("transformer"))] * self.num_hidden_layers

        super(Transformer, self).build(input_shape)

    def call(self, inputs, return_all_layers=False):
        input_tensro, attention_mask = inputs

        output_tensor = self.embedding_hidden_map(input_tensro)

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
    input_ids = tf.keras.layers.Input(shape=(512,))
    input_mask = tf.keras.layers.Input(shape=(512,))
    segment_ids = tf.keras.layers.Input(shape=(512,))

    config = AlbertConfig(max_position_embeddings=512, vocab_size=21128, type_vocab_size=2)
    config = AlbertConfig.from_json_file("/Users/lollipop/Downloads/albert_base/albert_config.json")
    albertModel = AlbertModel(config)
    output = albertModel((input_ids, input_mask, segment_ids))

    model = tf.keras.Model(inputs=(segment_ids, input_mask, input_ids), outputs=output)
    #
    # print(model.trainable_weights)
    model.summary()
    print(model.trainable_weights)
    for i in model.trainable_weights:
        print(i.name, i.shape)
    model.load_weights("/Users/lollipop/Documents/paper_coding/ALBert/out_new/bert_model.ckpt")
    print(model.trainable_weights)
    # model.load_weights('/Users/lollipop/Documents/paper_coding/Bert/out_new/bert_model.ckpt')
    # # print(model.trainable_weights)
    # # model.trainable_weights[-1].numpy = tf.random.uniform(shape=(768,), dtype=tf.float32)
    # # model.layers[-1].trainable_weights[-1].assign(tf.ones(shape=(768,), dtype=tf.float32))
    # # print(model.layers[-1].trainable_weights[-1])
    # print('@@@@@@@@')
    # print(model.trainable_weights)
    # for i in model.trainable_weights:
    #     print(i.name)
