"""绘图工具（Phase 5）：训练曲线 / 混淆矩阵 / 去噪对比 / 对照柱状图。

matplotlib 仅用于画图，不承担数据加载（红线2）。所有输出目录来自 config。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")           # 无 GUI 后端，脚本可安全保存图片
import matplotlib.pyplot as plt  # noqa: E402

import config


def _out(path):
    """相对路径按仓库根解析；确保父目录存在。"""
    p = path if os.path.isabs(path) else str(config.ROOT / path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def plot_train_curve(history, path=None, title="train curve"):
    """loss 与 acc 双曲线，横轴 epoch。

    history: dict(epoch=[], train_loss=[], train_acc=[], test_acc=[])
    """
    path = _out(path or os.path.join(config.OUTPUT_DIR, "train_curve.png"))
    ep = history["epoch"]
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax1.plot(ep, history["train_loss"], "o-", color="tab:red", label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(ep, history["train_acc"], "s-", color="tab:blue", label="train acc")
    ax2.plot(ep, history["test_acc"], "^--", color="tab:green", label="test acc")
    ax2.set_ylabel("accuracy", color="tab:blue")
    ax2.set_ylim(0.0, 1.02)

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=9)
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_confusion_matrix(cm, path=None, title="confusion matrix"):
    """10×10 混淆矩阵，行列 = 真实/预测标签，格内标数值，对角线为主。"""
    path = _out(path or os.path.join(config.OUTPUT_DIR, "confusion_matrix.png"))
    cm = np.asarray(cm)
    n = cm.shape[0]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(range(n)); ax.set_yticklabels(range(n))
    ax.set_title(title)
    thr = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > thr else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_denoise_triplet(clean, noisy, denoised, path=None, title="MNIST denoise"):
    """真实 MNIST 数字三栏对比：干净 / 加噪 / 去噪，标题标注与干净图的 MSE。

    如实呈现高斯去噪"抹平噪声"与"虚化边缘"的代价（不美化）。
    clean/noisy/denoised: (M, 28, 28) 序列。
    """
    path = _out(path or os.path.join(config.OUTPUT_DIR, "denoise_mnist_compare.png"))
    M = len(clean)
    fig, axes = plt.subplots(M, 3, figsize=(5.2, 1.6 * M))
    if M == 1:
        axes = axes[None, :]
    mse_n = float(np.mean((noisy - clean) ** 2))
    mse_d = float(np.mean((denoised - clean) ** 2))
    heads = ["clean", f"noisy  MSE={mse_n:.4f}", f"denoised  MSE={mse_d:.4f}"]
    for r in range(M):
        for c, img in enumerate((clean[r], noisy[r], denoised[r])):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(heads[c], fontsize=9)
            if c > 0:
                ax.set_ylabel(f"MSE {((img - clean[r]) ** 2).mean():.4f}", fontsize=7)
    fig.suptitle(f"{title}  (overall MSE: noisy={mse_n:.4f}, denoised={mse_d:.4f})", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_ablation_bar(labels, accs, path=None, title="denoise ablation: test accuracy"):
    """三组对照准确率柱状图 + 数值标注。"""
    path = _out(path or os.path.join(config.OUTPUT_DIR, "denoise_ablation.png"))
    accs = np.asarray(accs, dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    colors = ["tab:gray", "tab:green", "tab:red"][:len(labels)]
    bars = ax.bar(range(len(labels)), accs, color=colors, width=0.6)
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.01, f"{a*100:.2f}%",
                ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.0, 1.08)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
