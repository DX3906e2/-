"""CNN 组装 + 交叉熵损失（Phase 3，仅前向；反向留 Phase 4）。决策 D4/D11。

架构: 2×(Conv+ReLU+Pool) + Flatten + FC + Softmax，<100k 参数。
  (N,1,28,28) → Conv(1→8,3,same)→ReLU→Pool(2) → (N,8,14,14)
             → Conv(8→16,3,same)→ReLU→Pool(2) → (N,16,7,7)
             → Flatten → (N,784) → FC(784→10) → Softmax → (N,10)

损失并入本文件（D11），交叉熵用 log-sum-exp 数值稳定写法。
统一入口：从仓库根 `python src/cnn.py` 运行；反向/梯度检查入口已定义，Phase 4 实现。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import config
from layers import (Conv, ReLU, MaxPool, Flatten, FC, Softmax)


class CNN:
    def __init__(self, seed=None):
        if seed is None:
            seed = config.SEED
        rng = np.random.default_rng(seed)

        in_ch0 = 1
        c1, c2 = config.CONV_CHANNELS
        k = config.CONV_KERNEL
        self.conv1 = Conv(in_ch0, c1, k, rng)
        self.relu1 = ReLU()
        self.pool1 = MaxPool(config.POOL_SIZE)
        self.conv2 = Conv(c1, c2, k, rng)
        self.relu2 = ReLU()
        self.pool2 = MaxPool(config.POOL_SIZE)
        self.flatten = Flatten()
        # FC 输入维度仅由 config 推导：两次 2×2 池化后特征图 7×7
        feat = config.IMG_SIZE // (config.POOL_SIZE ** 2)
        fc_in = c2 * feat * feat
        self.fc = FC(fc_in, config.NUM_CLASSES, rng)
        self.softmax = Softmax()

        self.layers = [
            self.conv1, self.relu1, self.pool1,
            self.conv2, self.relu2, self.pool2,
            self.flatten, self.fc, self.softmax,
        ]

    def forward(self, x):
        """前向传播，返回 softmax 概率 (N, NUM_CLASSES)。"""
        h = np.asarray(x, dtype=np.float64)
        for layer in self.layers:
            h = layer.forward(h)
        return h

    def params(self):
        """返回所有可学习参数数组的扁平列表。"""
        ps = []
        for layer in self.layers:
            ps.extend(layer.params())
        return ps

    def _cross_entropy(self, logits, y):
        """log-sum-exp 数值稳定交叉熵，返回标量。logits: (N, C)，y: (N,) 整型。"""
        logits = np.asarray(logits, dtype=np.float64)
        shift = logits - logits.max(axis=1, keepdims=True)
        logsumexp = np.log(np.sum(np.exp(shift), axis=1)) + logits.max(axis=1)
        correct = logits[np.arange(logits.shape[0]), y]
        return float(np.mean(logsumexp - correct))

    def loss(self, x, y):
        """前向 + 交叉熵损失（标量）。"""
        self.forward(x)
        return self._cross_entropy(self.fc.out, y)

    def backward(self, y):
        """反向传播（须先 forward）。Softmax+交叉熵合并：dlogits = (probs - onehot) / N。

        loss 是 batch mean，故不得漏 1/N。逐层反向并填充各层 grad_W/grad_b。
        返回 dL/dx（(N,1,28,28)），一般无需使用。
        """
        y = np.asarray(y, dtype=np.int64)
        probs = self.softmax.probs
        N = y.shape[0]
        dlogits = probs.copy()
        dlogits[np.arange(N), y] -= 1.0
        dlogits /= N
        g = self.fc.backward(dlogits)
        g = self.flatten.backward(g)
        g = self.pool2.backward(g)
        g = self.relu2.backward(g)
        g = self.conv2.backward(g)
        g = self.pool1.backward(g)
        g = self.relu1.backward(g)
        g = self.conv1.backward(g)
        return g

    def grads(self):
        """返回与 params() 同序的梯度列表 [conv1.W/b, conv2.W/b, fc.W/b]。"""
        return [
            self.conv1.grad_W, self.conv1.grad_b,
            self.conv2.grad_W, self.conv2.grad_b,
            self.fc.grad_W, self.fc.grad_b,
        ]

    def gradient_check(self, x, y, eps=1e-6, noise=0.0, atol=1e-6, rtol=1e-4):
        """中心差分数值梯度 vs 解析梯度，逐参数组报告误差与判定。

        对全部 6 组参数（conv1.W/b, conv2.W/b, fc.W/b）逐一检查。建议小 batch（2~4）。
        返回 [(name, max_rel_error, atol_rtol_err, passed), ...]。

        判定准则（相对 + 绝对容差，二者互补）:
            |num - ana| <= atol + rtol * max(|num|, |ana|)
            atol_rtol_err = |num - ana| / (atol + rtol * max(|num|, |ana|))，< 1 为 PASS。

        为什么不用纯相对误差: 网络含 ReLU(0 处不可导) 与 MaxPool(块内并列时"赢家"跳变)
        两个 kink 源，有限差分在 kink 处误差天然偏大，且**eps 越大越严重**（实测
        eps 1e-3→1e-4→1e-5→1e-6 时误差单调收敛到 1e-8 量级）。故 eps 取小值 1e-6
        压制 kink；而小梯度元素(~1e-5)的绝对误差仅 ~1e-11，由 atol 放行。
        两个诉求（压 kink / 容小梯度）用 (小 eps + atol/rtol) 解耦，不再互相打架。
        需要时可将 noise>0 对输入/参数加微扰打破平局，检查后恢复参数。
        """
        x = np.asarray(x, dtype=np.float64)
        names = ["conv1.W", "conv1.b", "conv2.W", "conv2.b", "fc.W", "fc.b"]
        pgroups = [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, self.fc.W, self.fc.b]

        rng = np.random.default_rng(config.SEED + 12345)
        x_n = x + rng.uniform(-noise, noise, x.shape) if noise > 0 else x
        backups = [p.copy() for p in self.params()]
        if noise > 0:
            for p in self.params():
                p += rng.uniform(-noise, noise, p.shape)

        try:
            self.forward(x_n)
            self.backward(y)
            ana_groups = [np.asarray(g).copy() for g in self.grads()]
            results = []
            for name, p, ana in zip(names, pgroups, ana_groups):
                num = np.zeros_like(p)
                it = np.nditer(p, flags=["multi_index"])
                while not it.finished:
                    idx = it.multi_index
                    orig = p[idx]
                    p[idx] = orig + eps
                    lp = self.loss(x_n, y)
                    p[idx] = orig - eps
                    lm = self.loss(x_n, y)
                    p[idx] = orig
                    num[idx] = (lp - lm) / (2.0 * eps)
                    it.iternext()
                absdiff = np.abs(ana - num)
                rel = float((absdiff / np.maximum(np.abs(ana) + np.abs(num), 1e-8)).max())
                tol = atol + rtol * np.maximum(np.abs(ana), np.abs(num))
                norm = float((absdiff / tol).max())
                results.append((name, rel, norm, bool(norm < 1.0)))
        finally:
            for p, b in zip(self.params(), backups):
                p[...] = b  # 恢复参数
        return results


def _param_count(cnn):
    """返回各层参数量与总数。"""
    table = []
    total = 0
    for layer in cnn.layers:
        for p in layer.params():
            n = int(p.size)
            table.append((type(layer).__name__, p.shape, n))
            total += n
    return table, total


if __name__ == "__main__":
    cnn = CNN()
    x = np.random.default_rng(0).random((4, 1, config.IMG_SIZE, config.IMG_SIZE))
    y = np.array([0, 1, 2, 3])
    probs = cnn.forward(x)
    print("forward ->", probs.shape, "每行和:", probs.sum(axis=1))
    print("loss:", cnn.loss(x, y))
    table, total = _param_count(cnn)
    print("总参数量:", total)
