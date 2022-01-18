#! -*- coding:utf-8 -*-
# SuperGLUE评测
# MultiRC多项选择阅读理解任务
# 思路：段落, 问题，各个答案一起拼接后取[CLS]然后接Dense+Softmax分类

import json
from os import P_PGID
import numpy as np
from snippets import *
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from bert4keras.snippets import truncate_sequences
from tqdm import tqdm
from sklearn.metrics import f1_score

# 基本参数
labels = [False, True]
num_classes = len(labels)
maxlen = 128
batch_size = 32
epochs = 15


def load_data(filename):
    """加载数据
    格式：[(passage, question, answer, label)]
    """
    D = []
    with open(filename) as f:
        for i, l in enumerate(f):
            l = json.loads(l)
            p_all = l['passage']
            p = p_all['text']
            qs = p_all['questions']
            for trp1 in qs:
                q = trp1['question']
                ans = trp1['answers']
                for trp2 in ans:
                    an = trp2['text']
                    label = trp2.get('label', False)
                    D.append((p, q, an, label))
    return D


# 加载数据集
train_data = load_data(data_path + 'MultiRC/train.jsonl')
valid_data = load_data(data_path + 'MultiRC/val.jsonl')


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (p, q, an, label) in self.sample(random):
            p_ids = tokenizer.encode(p)[0]
            q_ids = tokenizer.encode(q)[0][1:]
            an_ids = tokenizer.encode(an)[0][1:]
            truncate_sequences(maxlen, -2, p_ids, q_ids, an_ids)
            token_ids = p_ids + q_ids + an_ids

            batch_token_ids.append(token_ids)
            batch_segment_ids.append([0] * len(token_ids))
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
    """保存验证集f1最好的模型
    """
    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_f1 = self.evaluate(valid_generator)
        if val_f1 > self.best_val_f1:
            self.best_val_acc = val_f1
            model.save_weights('weights/MultiRC.weights')
        
        print(
            u'val_f1: %.5f, best_val_f1: %.5f\n' %
            (val_f1, self.best_val_acc)
        )

    def evaluate(self, data):
        y_true_all = np.array([], dtype=int)
        y_pred_all = np.array([], dtype=int)
        for x_true, y_true in data:
            y_pred = model.predict(x_true).argmax(axis=1)
            y_true = y_true[:, 0]
            y_pred_all = np.append(y_pred_all, y_pred)
            y_true_all = np.append(y_true_all, y_true)
        f1 = f1_score(y_true_all,y_pred_all, average='micro')
        return f1
 

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
    fw = open(out_file, 'w')
    with open(in_file) as fr:
        for l, r in zip(fr, results):
            l = json.loads(l)
            p_all = l['passage']
            qs = p_all['questions']
            qs_out = []
            for trp1 in qs:
                ans = trp1['answers']
                ans_out = []
                for trp2 in ans:
                    ans_out.append({'idx': str(trp2['idx']), 'label':str(r)})
                qs_out.append({'idx': str(trp1['idx']), 'answers': ans_out})    
            l = json.dumps({'idx': str(l['idx']), 'passage':{'questions':qs_out}})
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
    model.load_weights('weights/MultiRC.weights')
    test_predict(
        in_file=data_path + 'MultiRC/test.jsonl',
        out_file='results/MultiRC.jsonl'
    )

else:

    model.load_weights('weights/MultiRC.weights')
