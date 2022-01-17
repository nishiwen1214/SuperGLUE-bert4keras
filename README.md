# 基于bert4keras的SuperGLUE基准代码

## 集齐CLUE，GLUE，SuperGLUE 三大自然语言理解榜单的基准代码
- [CLUE基准代码 (中文)](https://github.com/bojone/CLUE-bert4keras)
- [GLUE基准代码 (英文)](https://github.com/nishiwen1214/GLUE-bert4keras)
- [SuperGLUE基准代码 (英文)](https://github.com/nishiwen1214/SuperGLUE-bert4keras)

### ⭐️欢迎star和提问～

### 实验结果：

- test set:


### 使用
- 下载[SuperGLUE数据集](https://super.gluebenchmark.com/)和bert预训练的权重(这里使用的是[Google原版bert](https://github.com/google-research/bert))到指定文件夹；
- 例如：训练BoolQ，直接运行 `python BoolQ.py`。

### 环境
- 软件：bert4keras>=0.10.8, tensorflow = 1.15.0, keras = 2.3.1；
- 硬件：结果是用 RTX 3090 (24G)，注意3090显卡不支持Google的tf 1.x系列，需使用Nvidia的tf.15。安装教程可参考：https://www.bilibili.com/read/cv9162965/

### 更新
