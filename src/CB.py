#! -*- coding:utf-8 -*-
# SuperGLUE评测
# CB文本相似度
# 思路：premise和hypothesis拼接后取[CLS]然后接Dense+Softmax分类
# bert-base  val-F1: 78.57

import json
import numpy as np
from snippets import *
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score


# 基本参数
labels = ['entailment', 'neutral', 'contradiction']
num_classes = len(labels)
maxlen = 128
batch_size = 16
epochs = 10


def load_data(filename):
    """加载数据
    格式：[(premise, hypothesis, 标签id)]
    """
    D = []
    with open(filename) as f:
        for i, l in enumerate(f):
            l = json.loads(l)
            text1, text2, label = l['premise'], l['hypothesis'], l.get('label', 'entailment')
            D.append((text1, text2, labels.index(label)))
    return D


# 加载数据集
train_data = load_data(data_path + 'CB/train.jsonl')
valid_data = load_data(data_path + 'CB/val.jsonl')


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
        self.best_val_f1 = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc, f1 = self.evaluate(valid_generator)
        if f1 > self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights('weights/CB.weights')
        
        print(
            u'val_F1: %.5f, best_val_F1: %.5f, accuracy: %.5f\n' %
            (val_acc, self.best_val_f1, val_acc)
        )

    def evaluate(self, data):
        y_true_all = np.array([], dtype=int)
        y_pred_all = np.array([], dtype=int)
        for x_true, y_true in data:
            y_pred = model.predict(x_true).argmax(axis=1)
            y_true = y_true[:, 0]
            y_pred_all = np.append(y_pred_all, y_pred)
            y_true_all = np.append(y_true_all, y_true)
        acc = accuracy_score(y_true_all,y_pred_all)
        f1 = f1_score(y_true_all,y_pred_all, average='micro')
        return acc, f1
 

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
            l = json.dumps({'idx': str(l['idx']), 'label': labels[r]})
            fw.write(l + '\n')
    fw.close()


if __name__ == '__main__':
    model.load_weights('weights/CB.weights')
    evaluator = Evaluator()

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=1,
        callbacks=[evaluator]
    )

    model.load_weights('weights/CB.weights')
    test_predict(
        in_file=data_path + 'CB/test.jsonl',
        out_file='results/CB.jsonl'
    )

else:

    model.load_weights('weights/CB.weights')
