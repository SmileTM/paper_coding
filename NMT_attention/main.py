import tensorflow as tf
from modeling import Decoder, Encoder
from data_process import load_dataset, max_length, preprocess_sentence_en, preprocess_sentence_zh
from sklearn.model_selection import train_test_split
import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer = load_dataset('cmn.txt')
# 得到input，targe中的最大长度
max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)

input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor,
                                                                                                target_tensor,
                                                                                                test_size=0.2)
# 设置相关参数
BUFFER_SIZE = len(input_tensor_train)  # 设置buffer大小
BATCH_SIZE = 64  # 设置batch——size
steps_per_epoch = len(input_tensor_train) // BATCH_SIZE  # 得到训练集中每一个epoch中 batch的个数
embedding_dim = 256  # 设置embedding的输出维度
units = 1024  # 设置GRU 的输出维度，也就是GRU内部中 9W 的维度

vocab_inp_size = len(inp_lang_tokenizer.word_index) + 1
vocab_tar_size = len(targ_lang_tokenizer.word_index) + 1
# 得到 train数据集
dataset_train = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset_train = dataset_train.batch(BATCH_SIZE, drop_remainder=True)
# 得到 val数据集
dataset_val = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val)).shuffle(BUFFER_SIZE)
dataset_val = dataset_val.batch(BATCH_SIZE, drop_remainder=True)

# 定义encoder 和 decoder
encoder = Encoder(vocab_szie=vocab_inp_size, embedding_dim=embedding_dim, enc_units=units, batch_sz=BATCH_SIZE)
decoder = Decoder(vocab_size=vocab_tar_size, embedding_dim=embedding_dim, dec_units=units, batch_sz=BATCH_SIZE)

# 定义loss 和 optimizer
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

train_loss = tf.keras.metrics.Mean(name='train_loss')
val_loss = tf.keras.metrics.Mean(name='test_loss')


# 计算 去除mask后的loss 的平均值
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)

    loss_ *= mask

    return tf.reduce_mean(loss_)


# 定义checkpoint, checkpoint只保存模型参数
checkpoint_dir = './train_checkpoint'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


# 定义train_step
@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder((inp, enc_hidden))
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([targ_lang_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = decoder((dec_input, dec_hidden, enc_output))
            loss += loss_function(targ[:, t], predictions)
            # Teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    train_loss(batch_loss)


# 定义test_step
@tf.function
def test_step(inp, targ, enc_hidden):
    loss = 0
    enc_output, enc_hidden = encoder((inp, enc_hidden))
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

    for t in range(1, targ.shape[1]):
        predictions, dec_hidden, _ = decoder((dec_input, dec_hidden, enc_output))
        loss += loss_function(targ[:, t], predictions)
        # Teacher forcing
        predicted_id = tf.argmax(predictions, axis=-1)
        dec_input = tf.expand_dims(predicted_id, 1)

    loss = (loss / int(targ.shape[1]))

    val_loss(loss)


# 设置 tensorboard 文件保存的地址
current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tensorboard_path = './tensorboard_logs'
summary_writer_train = tf.summary.create_file_writer(os.path.join(tensorboard_path, 'train_' + current_time))
summary_writer_val = tf.summary.create_file_writer(os.path.join(tensorboard_path, 'val_' + current_time))
tf.summary.trace_on(profiler=True)  # 开启Graph跟踪


# 开始训练
EPOCHS = 100
for epoch in range(EPOCHS):
    start = time.time()
    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0
    for (batch, (inp, tar)) in enumerate(dataset_train.take(steps_per_epoch)):
        train_step(inp, tar, enc_hidden)

        if batch % 100 == 0:
            template = "Epoch {} Batch {} loss {:.4f} "
            print(template.format(epoch + 1, batch, train_loss.result()))

        with summary_writer_train.as_default():
            tf.summary.scalar('train_loss', train_loss.result(), step=epoch * steps_per_epoch + batch)

    for (inp, tar) in dataset_val:
        test_step(inp, tar, enc_hidden)

        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        with summary_writer_val.as_default():
            tf.summary.scalar('val_loss', val_loss.result(), step=epoch)

    print("Epoch {} loss {:4f} test_loss {:4f} ".format(epoch + 1, train_loss.result(), val_loss.result()))
    print("Time take for 1 epoch {} sec\n".format(time.time() - start))

    # 每个epoch后重置train_loss ,val_loss
    train_loss.reset_states()
    val_loss.reset_states()

# 保存可以部署的全模型pb文件
version = '1'
encoder.save('encoder_out/' + version)
decoder.save('decoder_out/' + version)


def evaluate(sentence):
    attention_plot = np.zeros((max_length_targ, max_length_inp))
    sentence = preprocess_sentence_zh(sentence)

    inputs = [inp_lang_tokenizer.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')

    inputs = tf.convert_to_tensor(inputs)
    result = ''
    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder((inputs, hidden))

    # 首次输入，用encoder的state
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang_tokenizer.word_index['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder((dec_input, dec_hidden, enc_out))

        attention_weights = tf.reshape(attention_weights, (-1,))

        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()
        result += targ_lang_tokenizer.index_word[predicted_id] + ' '

        if targ_lang_tokenizer.index_word[predicted_id] == '<end>':
            return result, sentence, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot


def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    cax = ax.matshow(attention, cmap='viridis')
    fig.colorbar(cax)

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def translate(sentence):
    result, sentence, attention_plot = evaluate(sentence)

    print("Input: %s" % (sentence))
    print("Predicted translation: {}".format(result))

    # 绘出注意力热力图
    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    plot_attention(attention_plot, sentence.split(' '), result.split(' '))


translate('我们去踢足球吧!')
