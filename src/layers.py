"""CNN 各层定义（Phase 3，仅前向；反向接口预留，Phase 4 实现）。

张量约定 NCHW: (N, C, H, W)，输入单通道 (N, 1, 28, 28)。
Conv 复用 filters.convolve2d（zero-pad / same，与 Phase 1 完全一致）。
参数初始化用 He 初始化 sqrt(2/fan_in)，由 config.SEED 驱动。
所有数值参数来自 config.py，本文件不出现魔法数字。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import config
from filters import convolve2d


def _he_scale(fan_in):
    """He 初始化标准差 sqrt(2/fan_in)。"""
    return np.sqrt(2.0 / fan_in)


class Conv:
    """2D 卷积层（NCHW）。权重 (out_ch, in_ch, k, k)，偏置 (out_ch,)。

    前向对每 (in_ch, out_ch) 通道对调用 filters.convolve2d（zero-pad same），
    再跨输入通道求和 + 偏置。反向接口预留（Phase 4）。
    """

    def __init__(self, in_ch, out_ch, k, rng):
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.k = k
        scale = _he_scale(in_ch * k * k)
        self.W = rng.standard_normal((out_ch, in_ch, k, k)) * scale
        self.b = np.zeros((out_ch,), dtype=np.float64)
        self.x = None

    def forward(self, x):
        x = np.asarray(x, dtype=np.float64)
        N, C, H, W = x.shape
        assert C == self.in_ch, f"Conv 输入通道 {C} != {self.in_ch}"
        out = np.zeros((N, self.out_ch, H, W), dtype=np.float64)
        for n in range(N):
            for o in range(self.out_ch):
                acc = np.zeros((H, W), dtype=np.float64)
                for c in range(self.in_ch):
                    acc = acc + convolve2d(x[n, c], self.W[o, c])
                out[n, o] = acc + self.b[o]
        self.x = x
        return out

    def backward(self, dout):
        # Phase 4 实现：依赖 filters.convolve2d_backward（zero-pad same）。
        raise NotImplementedError("Conv.backward 在 Phase 4 实现")

    def params(self):
        return [self.W, self.b]


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return np.where(self.mask, x, 0.0)

    def backward(self, dout):
        raise NotImplementedError("ReLU.backward 在 Phase 4 实现")

    def params(self):
        return []


class MaxPool:
    """2×2 / stride 2 最大池化（NCHW）。反向接口预留（Phase 4）。"""

    def __init__(self, size=config.POOL_SIZE):
        self.size = size

    def forward(self, x):
        x = np.asarray(x, dtype=np.float64)
        N, C, H, W = x.shape
        s = self.size
        assert H % s == 0 and W % s == 0, "MaxPool: 尺寸须被池化步长整除"
        out_h, out_w = H // s, W // s
        out = np.zeros((N, C, out_h, out_w), dtype=np.float64)
        self.argmax = np.zeros((N, C, out_h, out_w), dtype=int)
        for n in range(N):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        block = x[n, c, i * s:(i + 1) * s, j * s:(j + 1) * s]
                        flat = block.reshape(-1)
                        k = int(np.argmax(flat))
                        self.argmax[n, c, i, j] = k
                        out[n, c, i, j] = flat[k]
        self.x_shape = x.shape
        return out

    def backward(self, dout):
        raise NotImplementedError("MaxPool.backward 在 Phase 4 实现")

    def params(self):
        return []


class Flatten:
    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        raise NotImplementedError("Flatten.backward 在 Phase 4 实现")

    def params(self):
        return []


class FC:
    """全连接层 (N, in) -> (N, out)。He 初始化 fan_in = in。"""

    def __init__(self, in_dim, out_dim, rng):
        self.in_dim = in_dim
        self.out_dim = out_dim
        scale = _he_scale(in_dim)
        self.W = rng.standard_normal((out_dim, in_dim)) * scale
        self.b = np.zeros((out_dim,), dtype=np.float64)
        self.out = None

    def forward(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.out = x @ self.W.T + self.b
        return self.out

    def backward(self, dout):
        raise NotImplementedError("FC.backward 在 Phase 4 实现")

    def params(self):
        return [self.W, self.b]


class Softmax:
    """数值稳定 softmax（沿最后一轴），每行和为 1。反向接口预留（Phase 4）。"""

    def forward(self, x):
        x = np.asarray(x, dtype=np.float64)
        shift = x - x.max(axis=-1, keepdims=True)
        e = np.exp(shift)
        self.probs = e / e.sum(axis=-1, keepdims=True)
        return self.probs

    def backward(self, dout):
        raise NotImplementedError("Softmax.backward 在 Phase 4 实现")

    def params(self):
        return []
