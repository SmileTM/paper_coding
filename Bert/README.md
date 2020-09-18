# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
http://arxiv.org/abs/1810.04805

## 简介
* 支持Tensorflow2.0(keras、eager)
* 架构更加清晰
* 易于自定义
* 支持官方已发布的Tensorflow1.0 模型

## 使用
由于本代码是基于Tensorflow2.0构造的，使用Google已发布的Tensorflow1.0 版本的模型前，需提前转换。

### 模型转换

通过`tf1_ckpt_converter.py`脚本进行模型转换，产生支持本代码的同参数模型。
目前支持通过原生Google-bert代码构建的模型，如：
* Google原生bert: https://github.com/google-research/bert
* 哈工大版Chinese-BERT-wwm: https://github.com/ymcui/Chinese-BERT-wwm

总的来说，只要你是基于[Google原生bert](https://github.com/google-research/bert) ，未对代码中Bert架构内部进行修改的，均可转换。

### 例子
整体使用，就像使用正常的内部layer一样。
```python
    import tensorflow as tf
    import modeling

    assert tf.version.VERSION.startswith('2.')
    # 构建模型的输入
    input_ids = tf.keras.layers.Input(shape=(512,))
    input_mask = tf.keras.layers.Input(shape=(512,))
    segment_ids = tf.keras.layers.Input(shape=(512,))
    
    # 产生模型
    config = BertConfig.from_json_file('./bert_config.json')
    bertModel = BertModel(config)
    output = bertModel((input_ids, input_mask, segment_ids))
    
    model = tf.keras.Model(inputs=(segment_ids, input_mask, input_ids), outputs=output)
    model.summary()
    # 加载转换后的Bert权重 (不包括mlm_loss和nsp_loss部分)
    # 若需导入mlm_loss和nsp_loss部分，请使用 models.getPretrainingModel()
    model.load_weights('.out/bert_model.ckpt')

```



### Examples
包括 IMDB影评分类，知识蒸馏。
