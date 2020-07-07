# -*- coding: utf-8 -*-
#
# File: data_process.py
# Author: SmileTM
# Site: s-tm.cn
# Github: https://github.com/SmileTM
# Time: 07.05.2020
#

from tqdm import tqdm
from pathlib import Path
import numpy as np


# 合并文件
# 将IMDB数据集进行处理 初步合并处理
def IMDB_data_process(path):
    dataset = Path(path)
    for file in tqdm(dataset.rglob('*.txt')):
        with Path(file).open('r') as rfile, Path(file.parent.parent.name + '.' + file.parent.name).open('a') as wfile:
            for line in rfile.readlines():
                wfile.writelines(line.strip() + '\n')


# 再将 合并后的4个文件 处理为numpy矩阵
def get_dataset(tokenizer, max_seq_length):
    files_path = ['train.pos', 'train.neg', 'test.pos', 'test.neg']

    train_dataset = {'input_ids': [],
                     'input_mask': [],
                     'segment_ids': []}
    train_label = []

    test_dataset = {'input_ids': [],
                    'input_mask': [],
                    'segment_ids': []}
    test_label = []

    seq_length = max_seq_length - 2

    for file_path in files_path:

        input_ids = []
        input_mask = []
        segment_ids = []
        labels = []

        with Path(file_path).open('r') as file:
            for line in file.readlines():
                text = tokenizer.tokenize(line.strip())
                if len(text) > seq_length:
                    text = text[:seq_length]
                text = ["[CLS]"] + text + ["[SEP]"]

                ids = tokenizer.convert_tokens_to_ids(text)

                input_id = ids + [0] * (max_seq_length - len(ids))
                mask = [1] * len(ids) + [0] * (max_seq_length - len(ids))
                segment_id = [0] * max_seq_length  # 由于是情感分类 不需要上下句
                label = 1 if Path(file_path).suffix == '.pos' else 0

                input_ids.append(input_id)
                input_mask.append(mask)
                segment_ids.append(segment_id)
                labels.append(label)

        if Path(file_path).stem == 'train':
            train_dataset['input_ids'] += input_ids
            train_dataset['input_mask'] += input_mask
            train_dataset['segment_ids'] += segment_ids
            train_label += labels
        else:
            test_dataset['input_ids'] += input_ids
            test_dataset['input_mask'] += input_mask
            test_dataset['segment_ids'] += segment_ids
            test_label += labels

    train_dataset = {k: np.array(v) for k, v in train_dataset.items()}
    train_label = np.array(train_label)
    test_dataset = {k: np.array(v) for k, v in test_dataset.items()}
    test_label = np.array(test_label)

    return train_dataset, train_label, test_dataset, test_label


def get_hub_dataset(tokenizer, max_seq_length):
    files_path = ['train.pos', 'train.neg', 'test.pos', 'test.neg']

    train_dataset = {'input_word_ids': [],
                     'input_mask': [],
                     'segment_ids': []}
    train_label = []

    test_dataset = {'input_word_ids': [],
                    'input_mask': [],
                    'segment_ids': []}
    test_label = []

    seq_length = max_seq_length - 2

    for file_path in files_path:

        input_ids = []
        input_mask = []
        segment_ids = []
        labels = []

        with Path(file_path).open('r') as file:
            for line in file.readlines():
                text = tokenizer.tokenize(line.strip())
                if len(text) > seq_length:
                    text = text[:seq_length]
                text = ["[CLS]"] + text + ["[SEP]"]

                ids = tokenizer.convert_tokens_to_ids(text)

                input_id = ids + [0] * (max_seq_length - len(ids))
                mask = [1] * len(ids) + [0] * (max_seq_length - len(ids))
                segment_id = [0] * max_seq_length  # 由于是情感分类 不需要上下句
                label = 1 if Path(file_path).suffix == '.pos' else 0

                input_ids.append(input_id)
                input_mask.append(mask)
                segment_ids.append(segment_id)
                labels.append(label)

        if Path(file_path).stem == 'train':
            train_dataset['input_word_ids'] += input_ids
            train_dataset['input_mask'] += input_mask
            train_dataset['segment_ids'] += segment_ids
            train_label += labels
        else:
            test_dataset['input_word_ids'] += input_ids
            test_dataset['input_mask'] += input_mask
            test_dataset['segment_ids'] += segment_ids
            test_label += labels

    train_dataset = {k: np.array(v) for k, v in train_dataset.items()}
    train_label = np.array(train_label)
    test_dataset = {k: np.array(v) for k, v in test_dataset.items()}
    test_label = np.array(test_label)

    return train_dataset, train_label, test_dataset, test_label


if __name__ == '__main__':
    IMDB_data_process('./dataset')
