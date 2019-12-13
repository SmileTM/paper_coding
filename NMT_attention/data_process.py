import tensorflow as tf
import numpy as np
import re
import io

# 导入翻译句对
# http://www.manythings.org/anki/
path_to_file = 'cmn.txt'


# 对english进行处理
def preprocess_sentence_en(w):
    w = w.lower().strip()
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    # 返回删除字符尾部指定字符 后生成的新字符串
    w = w.rstrip().strip()
    w = '<start> ' + w + ' <end>'
    return w


# 对中文进行处理
def preprocess_sentence_zh(w):
    w = ' '.join(w)
    w = w.lower().strip()
    w = re.sub(r"([！？，。；：])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    # 返回删除字符尾部指定字符 后生成的新字符串
    w = w.rstrip().strip()
    w = '<start> ' + w + ' <end>'
    return w

# 对句对 进行处理， 返回 list-English， list-zh
def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence_en(l.split('\t')[0]), preprocess_sentence_zh(l.split('\t')[1])] for l in
                  lines[:num_examples]]
    # zip()相当于压缩， zip(*)相当于解压
    return zip(*word_pairs)


# 返回tensor中最长的长度
def max_length(tensor):
    return max(len(t) for t in tensor)


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, lang_tokenizer


# 加载数据 ，返回经过清洗后的数据，已经Tokenizer
def load_dataset(path, num_examples=None):
    targ_lang, inp_lang = create_dataset(path, num_examples)
    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer