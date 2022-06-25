#! -*- coding:utf-8 -*-
# SuperGLUE评测
# WSC代词消歧
# 思路：给span1、span2加上特殊标记，然后当成文本分类

import json
import numpy as np
from snippets import *
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from tqdm import tqdm
import re

# 基本参数
labels = [False, True]
num_classes = len(labels)
maxlen = 128
batch_size = 32
epochs = 10


def load_data(filename):
    """加载数据
    格式：[(文本, 标签id)]
    """
    D = []
    with open(filename) as f:
        for i, l in enumerate(f):
            l = json.loads(l)
            text, label = l['text'], labels.index(l.get('label', False))
            text_sp = re.split(' ', text)  
            s1 = l['target']['span1_index']
            e1 = s1 + len(re.split(' ', l['target']['span1_text'])) 
            s2 = l['target']['span2_index']
            e2 = s2 + len(re.split(' ',l['target']['span2_text']))
            text_new = ''
            if s1 < s2:
                text_sp.insert(s1,'|')
                text_sp.insert(e1+1,'|' )
                text_sp.insert(s2+2,'[')
                text_sp.insert(e2+3,']' )
                for w in text_sp:
                    text_new += w + ' '
            else:
                text_sp.insert(s2,'|')
                text_sp.insert(e2+1,'|' )
                text_sp.insert(s1+2,'[')
                text_sp.insert(e1+3,']' )
                for w in text_sp:
                    text_new += w + ' '
            D.append((text_new, label))
    return D


# 加载数据集
train_data = load_data(data_path + 'WSC/train.jsonl')
valid_data = load_data(data_path + 'WSC/val.jsonl')


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)

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


class Evaluator(keras.callbacks.Callback):
    """保存验证集acc最好的模型
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = self.evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('weights/WSC.weights')
        print(
            u'val_acc: %.5f, best_val_acc: %.5f\n' %
            (val_acc, self.best_val_acc)
        )

    def evaluate(self, data):
        total, right = 0., 0.
        for x_true, y_true in data:
            y_pred = model.predict(x_true).argmax(axis=1)
            y_true = y_true[:, 0]
            total += len(y_true)
            right += (y_true == y_pred).sum()
        return right / total


def test_predict(in_file, out_file):
    """输出测试结果到文件
    结果文件可以提交到 https://super.gluebenchmark.com/ 评测。
    """
    test_data = load_data(in_file)
    test_generator = data_generator(test_data, batch_size)

    results = []
    for x_true, _ in tqdm(test_generator, ncols=0):
        y_pred = model.predict(x_true).argmax(axis=1)
        results.extend(y_pred)
    labels_map = ['false','true']
    fw = open(out_file, 'w')
    with open(in_file) as fr:
        for l, r in zip(fr, results):
            l = json.loads(l)
            l = json.dumps({'idx': str(l['idx']), 'label': labels_map[r]})
            fw.write(l + '\n')
    fw.close()


if __name__ == '__main__':

    evaluator = Evaluator()

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )

    model.load_weights('weights/WSC.weights')

    test_predict(
        in_file=data_path + 'WSC/test.jsonl',
        out_file='results/WSC.jsonl'
    )

else:

    model.load_weights('weights/WSC.weights')
