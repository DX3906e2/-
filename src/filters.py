"""手写高斯核与二维卷积（禁止 scipy / cv2 / np.fft 等捷径）。

边界策略：zero-pad（补零），输出与输入同尺寸（'same'）。
该策略与 Phase 3 的 Conv 层保持一致。

所有数值参数只能从 config.py 读取，函数签名带默认参数；
本文件内不出现 5 / 1.2 这类魔法数字。
"""
import os
import sys

# 让同目录的 config.py 可被 import（不创建子包，仅临时扩展路径）。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402

import numpy as np  # noqa: E402
from numpy.lib.stride_tricks import sliding_window_view  # noqa: E402


def gaussian_kernel(k=None, sigma=None):
    """手写二维高斯核 (k, k)。归一化到和为 1，输出对称矩阵。

    公式: K[a, b] = exp(-(a^2 + b^2) / (2 * sigma^2))，再除以总和。
    """
    if k is None:
        k = config.GAUSS_K
    if sigma is None:
        sigma = config.GAUSS_SIGMA
    ax = np.arange(k, dtype=float) - (k - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax, indexing="ij")
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel


def _pad_zero(x, pad):
    """零填充（补零），x: (H, W)，pad: int 或 (ph, pw)。"""
    if isinstance(pad, int):
        ph = pw = pad
    else:
        ph, pw = pad
    return np.pad(x.astype(float), ((ph, ph), (pw, pw)), mode="constant")


def convolve2d_naive(image, kernel):
    """朴素多重循环版（基准，绝对正确）。边界：zero-pad / same。

    供 Phase 1 校验与生产版对齐；逻辑与 convolve2d 完全一致。
    """
    image = np.asarray(image, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = _pad_zero(image, (ph, pw))
    H, W = image.shape
    out = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            out[i, j] = np.sum(padded[i:i + kh, j:j + kw] * kernel)
    return out


def convolve2d(image, kernel):
    """向量化二维卷积（生产用）。边界：zero-pad / same，与 convolve2d_naive 同策略。

    使用 sliding_window_view 取局部窗口 + einsum 求和，不经过 scipy / cv2 / np.fft。
    """
    image = np.asarray(image, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = _pad_zero(image, (ph, pw))
    windows = sliding_window_view(padded, (kh, kw))
    return np.einsum("hwij,ij->hw", windows, kernel)


def _full_conv2d(a, b):
    """真卷积 a * b（零填充），输出 (Ha+Hb-1, Wa+Wb-1)。供 convolve2d_backward 使用。

    向量化实现：对 a 做 (Hb-1, Wb-1) 零填充，再用 sliding_window_view 取 (Hb, Wb)
    窗口与 b 做互相关，等价于"对 b 旋转 180° 后与 a 做互相关"的真卷积语义。
    输出尺寸 (Ha+Hb-1, Wa+Wb-1) 与 zero-padding 边界与原四重循环逐元素一致。
    仅用 np.pad + sliding_window_view + einsum，无 scipy / cv2 / np.fft。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    Hb, Wb = b.shape
    a_pad = np.pad(a, ((Hb - 1, Hb - 1), (Wb - 1, Wb - 1)), mode="constant")
    win = sliding_window_view(a_pad, (Hb, Wb))
    return np.einsum("mnrs,rs->mn", win, b)


def convolve2d_backward(dout, x, kernel, mode="zero"):
    """二维卷积反向传播。

    dout: (H, W) 上游梯度；x: (H, W) 输入；kernel: (kh, kw)。
    返回 (dx, dkernel)，与 Phase 3/4 的 Conv 层反向一致。

    dx = 真卷积(dout, kernel) 取中心 HxW；
    dkernel = 互相关(x_padded, dout)。
    """
    dout = np.asarray(dout, dtype=float)
    x = np.asarray(x, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2

    # dx：真卷积(dout, kernel) 再裁剪中心 HxW
    dx_full = _full_conv2d(dout, np.rot90(np.rot90(kernel)))
    dx = dx_full[ph:ph + x.shape[0], pw:pw + x.shape[1]]

    # dkernel：x 零填充后，与 dout 做互相关
    x_padded = _pad_zero(x, (ph, pw))
    x_win = sliding_window_view(x_padded, (kh, kw))
    dkernel = np.einsum("hwij,hw->ij", x_win, dout)
    return dx, dkernel
