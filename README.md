# naive-bayes-classifier

零依赖的**多项式朴素贝叶斯文本分类器**，纯 Python 标准库实现。
可从语料逐条/批量训练，也可直接吃「目录 = 类别」的语料目录；
模型可序列化为 JSON 持久化，预测时输出类别与各类对数概率。

## 功能简介

- 简单分词：英文小写单词 + 中文相邻字二元组；
- 拉普拉斯平滑（`alpha`，默认 1.0），避免零概率；
- `train(text, label)` 逐条学习 / `fit({类别: [文本...]})` 批量学习；
- `fit_from_directory(root)`：`root/<类别>/*.txt` 自动学习；
- `predict(text)` 返回 `(类别, {类别: logP})`；
- `save(path)` / `load(path)` 模型以 JSON 持久化。

## 快速开始

环境：Python 3.10+，零依赖。

```python
from naive_bayes import NaiveBayesClassifier

clf = NaiveBayesClassifier()
clf.fit({
    "体育": ["球队 进球 比赛 胜利", "篮球 足球 冠军"],
    "科技": ["机器学习 算法 数据 模型", "代码 软件 互联网"],
})
print(clf.predict("这个算法模型需要大量数据"))
clf.save("model.json")
```

命令行（从目录训练并预测）：

```bash
python3 naive_bayes.py --train-dir data --save model.json
python3 naive_bayes.py --load model.json --predict "球队又进球了"
```

## 使用示例（真实命令）

```bash
$ python3 naive_bayes.py --load model.json --predict "算法和模型需要训练数据"
预测类别：科技
  科技    logP=-18.231
  体育    logP=-24.775
```

## 无 API key 如何运行

本项目是纯本地的统计分类器，**不需要任何 API key**，也不联网。
用上面的代码或命令行即可完成训练与预测。

## 目录结构

```
naive-bayes-classifier/
├── naive_bayes.py              # 分类器 + CLI 入口
├── tests/
│   └── test_naive_bayes.py     # unittest（小样本验证可分性 + JSON 持久化）
├── README.md
├── LICENSE
└── .gitignore
```

## 运行测试

```bash
python3 -m unittest discover -s tests -v
```

## 许可证

[MIT](./LICENSE) © 2026 ljiang9
