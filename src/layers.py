"""CNN 各层定义（Phase 4：向量化前向 + 手写反向）。

张量约定 NCHW: (N, C, H, W)。
- Conv 前向：zero-pad 后 sliding_window_view + einsum 批量计算（无通道维循环）；
  反向复用 filters.convolve2d_backward 批量 NCHW 分支（zero-pad same，与 Phase 1/1.1 同策略）。
- 前向向量化：MaxPool 无 i/j 循环（reshape/transpose + max，保存 argmax）。
- 反向：Softmax+交叉熵合并（在 cnn.py，含 1/N）；FC/Flatten/MaxPool/ReLU/Conv 各自手写。
参数初始化 He（sqrt(2/fan_in)），由 config.SEED 驱动。参数只来自 config.py。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import config
from filters import convolve2d_backward


def _he_scale(fan_in):
    """He 初始化标准差 sqrt(2/fan_in)。"""
    return np.sqrt(2.0 / fan_in)


class Conv:
    """2D 卷积层（NCHW）。权重 (out_ch, in_ch, k, k)，偏置 (out_ch,)。"""

    def __init__(self, in_ch, out_ch, k, rng):
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.k = k
        scale = _he_scale(in_ch * k * k)
        self.W = rng.standard_normal((out_ch, in_ch, k, k)) * scale
        self.b = np.zeros((out_ch,), dtype=np.float64)
        self.x = None
        self.grad_W = None
        self.grad_b = None

    def forward(self, x):
        x = np.asarray(x, dtype=np.float64)
        N, C, H, W = x.shape
        assert C == self.in_ch, f"Conv 输入通道 {C} != {self.in_ch}"
        k = self.k
        pad = k // 2
        x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
        win = sliding_window_view(x_pad, (k, k), axis=(-2, -1))  # (N,C,H,W,k,k)
        out = np.einsum("nchwij,ocij->nohw", win, self.W) + self.b[None, :, None, None]
        self.x = x
        return out

    def backward(self, dout):
        dout = np.asarray(dout, dtype=np.float64)
        dx, dW = convolve2d_backward(dout, self.x, self.W)  # zero-pad same（filters.py）
        self.grad_W = dW
        self.grad_b = dout.sum(axis=(0, 2, 3))  # db[o] = dout[:, o].sum()
        return dx

    def params(self):
        return [self.W, self.b]


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return np.where(self.mask, x, 0.0)

    def backward(self, dout):
        return dout * self.mask

    def params(self):
        return []


class MaxPool:
    """2×2 / stride 2 最大池化（NCHW）。前向无 i/j 循环，保存 argmax 供反向。"""

    def __init__(self, size=config.POOL_SIZE):
        self.size = size

    def forward(self, x):
        x = np.asarray(x, dtype=np.float64)
        N, C, H, W = x.shape
        s = self.size
        assert H % s == 0 and W % s == 0, "MaxPool: 尺寸须被池化步长整除"
        oh, ow = H // s, W // s
        xr = x.reshape(N, C, oh, s, ow, s)
        xt = xr.transpose(0, 1, 2, 4, 3, 5)              # (N,C,oh,ow,s,s)
        xf = xt.reshape(N, C, oh, ow, s * s)
        self.argmax = xf.argmax(axis=-1)                  # 每块赢家（平局取首，与循环版一致）
        self.x_shape = x.shape
        return xf.max(axis=-1)

    def backward(self, dout):
        dout = np.asarray(dout, dtype=np.float64)
        N, C, oh, ow = dout.shape
        s = self.size
        H, W = self.x_shape[2], self.x_shape[3]
        dx = np.zeros((N, C, H, W), dtype=np.float64)
        rows = self.argmax // s                           # 赢家在块内的行
        cols = self.argmax % s                            # 赢家在块内的列
        n_idx, c_idx, i_idx, j_idx = np.indices((N, C, oh, ow))
        dx[n_idx, c_idx, i_idx * s + rows, j_idx * s + cols] = dout  # 只归赢家，不平均
        return dx

    def params(self):
        return []


class Flatten:
    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.shape)

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
        self.x = None
        self.out = None
        self.grad_W = None
        self.grad_b = None

    def forward(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.x = x
        self.out = x @ self.W.T + self.b
        return self.out

    def backward(self, dout):
        dout = np.asarray(dout, dtype=np.float64)
        self.grad_W = dout.T @ self.x
        self.grad_b = dout.sum(axis=0)
        return dout @ self.W

    def params(self):
        return [self.W, self.b]


class Softmax:
    """数值稳定 softmax（沿最后一轴），每行和为 1。

    注：与交叉熵联合训练时，dlogits = (probs - onehot) / N 在 cnn.py 合并计算；
    本 backward 保留通用 softmax 反向（dout 非损失直接梯度时使用）。
    """

    def forward(self, x):
        x = np.asarray(x, dtype=np.float64)
        shift = x - x.max(axis=-1, keepdims=True)
        e = np.exp(shift)
        self.probs = e / e.sum(axis=-1, keepdims=True)
        return self.probs

    def backward(self, dout):
        p = self.probs
        return p * (dout - (dout * p).sum(axis=-1, keepdims=True))

    def params(self):
        return []
