import tensorflow as tf


class Attention(tf.keras.layers.Layer):
    def __init__(self, d_model, n_head):
        super(Attention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qw = tf.keras.layers.Dense(d_model)
        self.kw = tf.keras.layers.Dense(d_model)
        self.vw = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, input, batch_size, n_head, d_head):
        '''
        b: bath_size
        l: tensor_len
        n: the number of head
        d: the dimension of every head
        '''
        input = tf.reshape(input, (batch_size, -1, n_head, d_head))
        return tf.einsum('blnd->bnld', input)

    def attention_procedure(self, q, k, v, mask):
        qk = tf.einsum('bnld,bnLd->bnlL', q, k)
        dk = tf.cast(k.shape[-1], qk.dtype)
        attention_weights = qk / tf.sqrt(dk)

        if mask is not None:
            attention_weights += ((1 - mask) * -1e9)
        attention_weights = tf.nn.softmax(attention_weights, axis=-1)

        out = tf.einsum('bnlL,bnLd->bnld', attention_weights, v)
        return out, attention_weights

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape=(self.n_head, self.d_head, self.d_model), trainable=True,
                                      dtype=tf.float32)

    def call(self, inputs):
        q, k, v, mask = inputs
        batch_size = tf.shape(q)[0]
        q = self.qw(q)
        k = self.qw(k)
        v = self.qw(v)

        q = self.split_heads(q, batch_size, n_head=self.n_head, d_head=self.d_head)
        k = self.split_heads(k, batch_size, n_head=self.n_head, d_head=self.d_head)
        v = self.split_heads(v, batch_size, n_head=self.n_head, d_head=self.d_head)

        out, attention_weights = self.attention_procedure(q, k, v, mask)

        # out = tf.einsum('bnld->blnd', out)
        #
        # # out = tf.einsum('blnd,ndm->blm', out, self.kernel)

        return out, attention_weights


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, n_head, dropout_lr=0.1):
        super(TransformerBlock, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.attention = Attention(d_model, n_head)
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_lr)
        self.dropout2 = tf.keras.layers.Dropout(dropout_lr)
        self.ffn = tf.keras.layers.Dense(d_model)

    def call(self, inputs):
        # x : input_sentence
        x, mask = inputs
        out1, attention_weights = self.attention((x, x, x, mask))

        out1 = self.dropout1(out1)

        out1 = self.layer_norm1(out1 + x)

        out2 = self.ffn(out1)
        out2 = self.dropout2(out2)
        out2 = self.layer_norm2(out2 + out1)

        return out2


if __name__ == '__main__':
    att = Attention(d_model=768, n_head=12)
    input = tf.keras.layers.Input(shape=(512, 768))
    out = att([input, input, input, None])
    model = tf.keras.models.Model(inputs=input, outputs=out)
    model.summary()
    print(model.layers)
