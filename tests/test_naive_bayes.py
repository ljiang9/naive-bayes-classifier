import json
import os
import tempfile
import unittest

from naive_bayes import NaiveBayesClassifier, fit_from_directory, tokenize


class TestTokenize(unittest.TestCase):
    def test_mixed(self):
        toks = tokenize("I love 自然语言")
        self.assertIn("love", toks)
        self.assertIn("自然", toks)
        self.assertIn("然语", toks)


class TestClassify(unittest.TestCase):
    def setUp(self):
        self.clf = NaiveBayesClassifier()
        self.clf.fit({
            "体育": [
                "球队赢得了比赛 比分 进球 胜利",
                "篮球 足球 运动员 比赛 冠军",
                "球迷 体育场 进球 比赛 胜利",
            ],
            "科技": [
                "机器学习 算法 数据 模型 神经网络",
                "程序员 代码 开源 软件 互联网",
                "人工智能 芯片 算法 数据 模型",
            ],
        })

    def test_separable_sports(self):
        label, _ = self.clf.predict("球队又进了一个球，赢得比赛")
        self.assertEqual(label, "体育")

    def test_separable_tech(self):
        label, _ = self.clf.predict("这个算法模型需要大量数据训练")
        self.assertEqual(label, "科技")

    def test_scores_distinguish(self):
        scores = self.clf.predict_scores("程序员写代码")
        self.assertGreater(scores["科技"], scores["体育"])


class TestPersistence(unittest.TestCase):
    def test_save_load_roundtrip(self):
        clf = NaiveBayesClassifier(alpha=1.0)
        clf.fit({"正": ["好 棒 优秀", "愉快 满意"], "负": ["差 烂 糟糕", "失望 生气"]})
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.json")
            clf.save(path)
            self.assertTrue(os.path.exists(path))
            back = NaiveBayesClassifier.load(path)
            label, _ = back.predict("这个产品很棒很满意")
            self.assertEqual(label, "正")


class TestFitDirectory(unittest.TestCase):
    def test_from_dir(self):
        with tempfile.TemporaryDirectory() as d:
            for cat, files in {
                "catA": ["alpha 样本 一", "alpha 样本 二"],
                "catB": ["beta 样本 甲", "beta 样本 乙"],
            }.items():
                cdir = os.path.join(d, cat)
                os.makedirs(cdir)
                for i, t in enumerate(files):
                    with open(os.path.join(cdir, f"{i}.txt"), "w", encoding="utf-8") as f:
                        f.write(t)
            clf = fit_from_directory(d)
            label, _ = clf.predict("alpha 相关内容")
            self.assertEqual(label, "catA")


if __name__ == "__main__":
    unittest.main()
