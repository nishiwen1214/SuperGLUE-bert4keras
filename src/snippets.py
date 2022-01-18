#! -*- coding: utf-8 -*-
# SuperGLUE评测
# 模型配置文件

import os
from bert4keras.backend import keras, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.optimizers import extend_with_gradient_accumulation
import os
# 选择使用第几张GPU卡，'0'为第一张
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# 通用参数
data_path = '../Datasets/SuperGLUE/'
learning_rate = 1e-5
pooling = 'first'

# 权重目录
if not os.path.exists('weights'):
    os.mkdir('weights')

# 输出目录
if not os.path.exists('results'):
    os.mkdir('results')

# 预训练模型路径
# bert-base-uncased
# config_path = '../Pre-trained_models/uncased_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = '../Pre-trained_models/uncased_L-12_H-768_A-12/bert_model.ckpt'
# dict_path = '../Pre-trained_models/uncased_L-12_H-768_A-12/vocab.txt'

# bert-large-cased
config_path = '../Pre-trained_models/cased_L-24_H-1024_A-16/bert_config.json'
checkpoint_path = '../Pre-trained_models/cased_L-24_H-1024_A-16/bert_model.ckpt'
dict_path = '../Pre-trained_models/cased_L-24_H-1024_A-16/vocab.txt'


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 预训练模型
base = build_transformer_model(
    config_path, checkpoint_path, application='unilm', return_keras_model=False
)

# 模型参数
last_layer = 'Transformer-%s-FeedForward-Norm' % (base.num_hidden_layers - 1)

if pooling == 'first':
    pooling_layer = keras.layers.Lambda(lambda x: x[:, 0])
elif pooling == 'avg':
    pooling_layer = keras.layers.GlobalAveragePooling1D()
elif pooling == 'max':
    pooling_layer = keras.layers.GlobalMaxPooling1D()

# 优化器
Adam = extend_with_gradient_accumulation(Adam, name='Adam')

optimizer = Adam(
    learning_rate=learning_rate
)

optimizer2 = Adam(
    learning_rate=learning_rate,
    grad_accum_steps=2
)

optimizer4 = Adam(
    learning_rate=learning_rate,
    grad_accum_steps=4
)

