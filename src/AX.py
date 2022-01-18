#! -*- coding:utf-8 -*-
# SuperGLUE评测
# AX-b and AX-g
# 思路：基于训练好的RTE模型进行预测

import json
import numpy as np
from six import b
from snippets import *
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from tqdm import tqdm


# 基本参数
labels = ['entailment', 'not_entailment']
num_classes = len(labels)
maxlen = 128
batch_size = 32
epochs = 10


def load_data(filename, data):
    """加载数据
    格式：[(premise, hypothesis, 标签id)]
    """
    D = []
    with open(filename) as f:
        for i, l in enumerate(f):
            l = json.loads(l)
            if data == 'b':
                text1, text2, label = l['sentence1'], l['sentence2'], l.get('label', 'entailment')
            else:   
                text1, text2, label = l['premise'], l['hypothesis'], l.get('label', 'entailment')
            D.append((text1, text2, labels.index(label)))
    return D


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text1, text2, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(
                text1, text2, maxlen=maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append([0] * len(segment_ids))
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 构建模型
output = base.model.get_layer(last_layer).output
output = pooling_layer(output)
output = keras.layers.Dense(
    units=num_classes,
    activation='softmax',
    kernel_initializer=base.initializer
)(output)

model = keras.models.Model(base.model.input, output)
model.summary()

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

 

def test_predict(in_file, out_file, data='b'):
    """输出测试结果到文件
    结果文件可以提交到 https://super.gluebenchmark.com/ 评测。
    """
    test_data = load_data(in_file, data)
    test_generator = data_generator(test_data, batch_size)

    results = []
    for x_true, _ in tqdm(test_generator, ncols=0):
        y_pred = model.predict(x_true).argmax(axis=1)
        results.extend(y_pred)
    fw = open(out_file, 'w')
    with open(in_file) as fr:
        for l, r in zip(fr, results):
            l = json.loads(l)
            l = json.dumps({'idx': str(l['idx']), 'label': labels[r]})
            fw.write(l + '\n')
    fw.close()


if __name__ == '__main__':

    model.load_weights('weights/RTE.weights')

    test_predict(
        in_file=data_path + 'AX-b/AX-b.jsonl',
        out_file='results/AX-b.jsonl',
        data='b'
    )

    test_predict(
        in_file=data_path + 'AX-g/AX-g.jsonl',
        out_file='results/AX-g.jsonl',
        data='g'
    )

