# -*- coding: utf-8 -*-
#
# File: PTDemo.py
# Author: SmileTM
# Site: s-tm.cn
# Github: https://github.com/SmileTM
# Time: 06.10.2023
#
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

import os


# PyTorch 使用HuggingFace transformers保存模型方式

class ModelConfig(PretrainedConfig):
    model_type = "ptmodel"

    def __init__(self, size=1024, **kwargs):
        super().__init__(**kwargs)
        self.size = size


class Model(PreTrainedModel):
    def __init__(self, config: ModelConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config
        self.liner1 = torch.nn.Linear(self.config.size, self.config.size)
        self.liner2 = torch.nn.Linear(self.config.size, self.config.size)

    def forward(self, input):
        out = self.liner1(input)
        out = self.liner2(out)
        return out


if __name__ == '__main__':
    saved_path = "saved_pt"
    os.makedirs(saved_path, exist_ok=True)

    config = ModelConfig(size=1024)

    model = Model(config=config)
    data = torch.ones((1024, 1024))
    out = model(data)
    print(f"model output:", out)
    model.save_pretrained(saved_path, max_shard_size="6MB")
    # 打印model参数
    print(model.state_dict())

    model1 = Model(config=config).from_pretrained(saved_path, from_pt=True, config=config)
    out1 = model1(data)
    print(f"model1 output:", out)
    # 打印model1 参数
    print(model1.state_dict())

    '''
    >>> tree saved_pt 
    
    saved_pt
    ├── config.json
    ├── pytorch_model-00001-of-00002.bin
    ├── pytorch_model-00002-of-00002.bin
    └── pytorch_model.bin.index.json

    '''
