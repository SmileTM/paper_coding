# IMDB 影评分类（二分类）

## IMDB数据集
IMDB影评数据集中的训练集和测试集分别位于 train 和 test 两个目录，每个目 录下均有 pos 和 neg 两个子目录，分别代表正面评价的文本和负面评价的文本;每一个训练样例一个文件，文本命名方式为:id_rating.txt，其 中 id 为样例唯一 id，rating 为该文本的情感极性评分，正面评价为 7-10 分，负面评价为 0-4 分。

其中，train和test数据集各有25000个。

[数据集](https://ai.stanford.edu/~amaas/data/sentiment/)



## 使用

### First
bert 原生**TF1**预训练模型下载 https://github.com/google-research/bert
下载后 通过'tf1_ckpt_converter.py' 将TF1模型权重 进行转换 .

### Second
对IMDB数据进行处理，生成 'train.pos', 'train.neg', 'test.pos', 'test.neg' 4个文件
```python
python data_process.py
```

### Third
在引入模型,并在数据集上进行微调.
```python
python train.py
```


## 结果

在使用Bert预训练语言模型 设置BATCH_SIZE=16， 在经过3个epoch后， 验证集上的准确率达到了 94.12% 。
```python
Epoch 1/3
1562/1562 [==============================] - 4014s 3s/step - loss: 0.2769 - acc: 0.8814 - val_loss: 0.1815 - val_acc: 0.9299
Epoch 2/3
1562/1562 [==============================] - 4006s 3s/step - loss: 0.1166 - acc: 0.9584 - val_loss: 0.1704 - val_acc: 0.9386
Epoch 3/3
1562/1562 [==============================] - 4005s 3s/step - loss: 0.0318 - acc: 0.9914 - val_loss: 0.2132 - val_acc: 0.9412
```
