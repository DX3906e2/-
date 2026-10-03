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

    def gradient_check(self, x, y, eps=1e-5):
        """梯度检查入口（Phase 4 实现，需 backward）。本阶段暂不可用。"""
        raise NotImplementedError("gradient_check 在 Phase 4 实现（依赖各层 backward）")


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
