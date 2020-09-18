# Bert模型蒸馏

这里student-model 使用了4层 transformer encoder block。

训练方法与[tiny-Bert](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT) 方法一样。

首先，对Embedding 层 和 teacher model 每3层的transformer encoder block 进行蒸馏，蒸馏中使用的是 MSE 损失函数。
然后，在对最后的输出结果进行进一步整体蒸馏，蒸馏过程中选取合适的 蒸馏温度系数。

在Teacher model 在IMDB 上的成绩：

```python
train set accuracy: 99.916%   eval set accuracy: 93.424%
```

通过蒸馏，student model 在IMDB 上的成绩：

```python
train set accuracy: 99.176%   eval set accuracy: 83.968%

```
