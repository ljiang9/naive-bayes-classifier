#!/usr/bin/env python3
"""naive_bayes —— 多项式朴素贝叶斯文本分类器（零第三方依赖）。

支持：
  - 逐条 train(text, label) 学习，或 fit(label -> [text, ...]) 批量学习；
  - 从目录学习：目录下每个子目录名即类别，里面的 .txt 即该类样本；
  - 模型持久化到 JSON（save / load）；
  - predict 给出预测类别与各类别对数概率。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict

_CJK = re.compile(r"[\u4e00-\u9fff]")
_WORD = re.compile(r"[a-z0-9][a-z0-9]*")


def tokenize(text: str) -> list[str]:
    toks = _WORD.findall(text.lower())
    buf: list[str] = []

    def flush():
        for i in range(len(buf) - 1):
            toks.append(buf[i] + buf[i + 1])
        buf.clear()

    for ch in text:
        if _CJK.match(ch):
            buf.append(ch)
        else:
            flush()
    flush()
    return toks


class NaiveBayesClassifier:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.class_doc_count: dict[str, int] = defaultdict(int)
        self.class_token_count: dict[str, int] = defaultdict(int)
        self.word_count: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.vocab: set[str] = set()

    def train(self, text: str, label: str) -> None:
        self.class_doc_count[label] += 1
        for tok in tokenize(text):
            self.word_count[label][tok] += 1
            self.class_token_count[label] += 1
            self.vocab.add(tok)

    def fit(self, labeled: dict[str, list[str]]) -> None:
        for label, texts in labeled.items():
            for t in texts:
                self.train(t, label)

    @property
    def _total_docs(self) -> int:
        return sum(self.class_doc_count.values())

    def _log_prior(self, label: str) -> float:
        return math.log(self.class_doc_count[label] / self._total_docs)

    def _log_likelihood(self, token: str, label: str) -> float:
        cnt = self.word_count[label].get(token, 0)
        denom = self.class_token_count[label] + self.alpha * len(self.vocab)
        return math.log((cnt + self.alpha) / denom)

    def predict_scores(self, text: str) -> dict[str, float]:
        scores: dict[str, float] = {}
        toks = tokenize(text)
        for label in self.class_doc_count:
            s = self._log_prior(label)
            for tok in toks:
                s += self._log_likelihood(tok, label)
            scores[label] = s
        return scores

    def predict(self, text: str) -> tuple[str, dict[str, float]]:
        scores = self.predict_scores(text)
        if not scores:
            raise ValueError("模型尚未训练任何类别")
        best = max(scores, key=scores.get)
        return best, scores

    def save(self, path: str) -> None:
        data = {
            "alpha": self.alpha,
            "class_doc_count": dict(self.class_doc_count),
            "class_token_count": dict(self.class_token_count),
            "word_count": {c: dict(w) for c, w in self.word_count.items()},
            "vocab": sorted(self.vocab),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "NaiveBayesClassifier":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        obj = cls(alpha=data["alpha"])
        obj.class_doc_count = defaultdict(int, data["class_doc_count"])
        obj.class_token_count = defaultdict(int, data["class_token_count"])
        for c, w in data["word_count"].items():
            obj.word_count[c] = defaultdict(int, w)
        obj.vocab = set(data["vocab"])
        return obj


def fit_from_directory(root: str) -> NaiveBayesClassifier:
    clf = NaiveBayesClassifier()
    for label in sorted(os.listdir(root)):
        cdir = os.path.join(root, label)
        if not os.path.isdir(cdir):
            continue
        for fn in sorted(os.listdir(cdir)):
            if not fn.endswith(".txt"):
                continue
            with open(os.path.join(cdir, fn), "r", encoding="utf-8") as f:
                clf.train(f.read(), label)
    return clf


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="朴素贝叶斯文本分类器")
    p.add_argument("--train-dir", help="按子目录类别训练")
    p.add_argument("--save", help="训练后把模型保存到该 JSON")
    p.add_argument("--load", help="加载已有模型 JSON")
    p.add_argument("--predict", help="对这段文本预测类别")
    args = p.parse_args(argv)
    if args.load:
        clf = NaiveBayesClassifier.load(args.load)
    elif args.train_dir:
        clf = fit_from_directory(args.train_dir)
        if args.save:
            clf.save(args.save)
            print(f"模型已保存到 {args.save}，类别：{list(clf.class_doc_count)}")
    else:
        print("请提供 --train-dir 或 --load", file=sys.stderr)
        return 2
    if args.predict is not None:
        label, scores = clf.predict(args.predict)
        print(f"预测类别：{label}")
        for c, s in sorted(scores.items(), key=lambda kv: -kv[1]):
            print(f"  {c}\tlogP={s:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
