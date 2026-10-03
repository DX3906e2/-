"""CLI 训练（Phase 5）：真实去噪管线 + 训练循环 + SGD + 存权重 + 训练曲线。

管线（D9 硬约束，禁止装饰性跳过/回退）：
    干净图 → 加高斯噪声(sigma=config.NOISE_SIGMA) → 高斯核真实二维卷积去噪
           → （可选关闭噪声/去噪构成三组对照）→ CNN

三组对照（相同 seed / 超参 / 数据切分）：
    A 基准  --no-noise --no-denoise      干净→干净
    B 主线  （默认）                     加噪+去噪 → 加噪+去噪
    C 消融  --no-denoise                 加噪 → 加噪（不去噪）

用法：
    python src/train.py
    python src/train.py --n-train 1000 --epochs 2
    python src/train.py --no-noise --no-denoise --tag A
    python src/train.py --no-denoise --tag C
参数全部来自 config.py（CLI 覆盖除外）；未引入任何新依赖（仅 argparse/numpy/matplotlib）。
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import config
import data
import viz
from cnn import CNN
from filters import gaussian_kernel, convolve2d_batch

# 与 CNN.params() 同序的权重键
PARAM_KEYS = ["conv1_W", "conv1_b", "conv2_W", "conv2_b", "fc_W", "fc_b"]
_EVAL_SUBSET = 2000     # 每 epoch 曲线用的测试子集（末轮用全集做最终准确率）
_LOSS_SUBSET = 256      # 每 epoch 曲线用的 train loss 子集（小批量，控内存）


def _root(path):
    p = path if os.path.isabs(path) else str(config.ROOT / path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def denoise_batch(X, kernel):
    """对 (N,28,28) 批量做真实高斯卷积去噪（复用 filters.convolve2d_batch，zero-pad same）。"""
    return convolve2d_batch(X, kernel)


def make_inputs(X_clean, kernel, use_noise, use_denoise, rng, sigma=None):
    """按组别把干净图变成模型输入：加噪(可选) → 去噪(可选)。真实执行，不跳步。"""
    if sigma is None:
        sigma = config.NOISE_SIGMA
    X = X_clean
    if use_noise:
        X = data.add_gaussian_noise(X, sigma, rng=rng)
    if use_denoise:
        X = denoise_batch(X, kernel)
    return X


def save_model(cnn, path):
    p = _root(path)
    np.savez(p, **{k: v for k, v in zip(PARAM_KEYS, cnn.params())})
    return p


def load_model(cnn, path):
    p = path if os.path.isabs(path) else str(config.ROOT / path)
    with np.load(p) as f:
        for k, tgt in zip(PARAM_KEYS, cnn.params()):
            tgt[...] = f[k]
    return cnn


def predict_acc(cnn, X, y, batch_size):
    """在 (N,28,28) 干净图上评测准确率（内部统一 to_nchw）。"""
    correct = 0
    for i in range(0, len(X), batch_size):
        probs = cnn.forward(data.to_nchw(X[i:i + batch_size]))
        correct += int((probs.argmax(axis=1) == y[i:i + batch_size]).sum())
    return correct / len(X)


def confusion(cnn, X, y, batch_size, n_classes=None):
    n_classes = n_classes or config.NUM_CLASSES
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for i in range(0, len(X), batch_size):
        pred = cnn.forward(data.to_nchw(X[i:i + batch_size])).argmax(axis=1)
        for t, p in zip(y[i:i + batch_size], pred):
            cm[int(t), int(p)] += 1
    return cm


def run(n_train=None, epochs=None, lr=None, batch_size=None,
        use_noise=True, use_denoise=True, seed=None, tag="main",
        verbose=True):
    """训练一次，返回 dict(history=..., test_acc=..., test_acc_subset=..., model_path=...)。"""
    n_train = config.N_TRAIN if n_train is None else n_train
    epochs = config.EPOCHS if epochs is None else epochs
    lr = config.LR if lr is None else lr
    batch_size = config.BATCH_SIZE if batch_size is None else batch_size
    seed = config.SEED if seed is None else seed
    sfx = "" if tag == "main" else f"_{tag}"

    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    log(f"===== train[{tag}] n_train={n_train} epochs={epochs} lr={lr} "
        f"batch={batch_size} noise={use_noise} denoise={use_denoise} seed={seed} =====")

    Xtr, ytr, Xte, yte = data.load_mnist(n_train=n_train, log=(lambda *a: None))
    kernel = gaussian_kernel()                       # config.GAUSS_K / GAUSS_SIGMA

    # 测试输入：噪声固定（用独立 rng），保证各 epoch / evaluate 可比可复现
    Xte_in = make_inputs(Xte, kernel, use_noise, use_denoise,
                         rng=np.random.default_rng(seed + 1))

    cnn = CNN(seed=seed)
    rng = np.random.default_rng(seed)                # 训练噪声逐 epoch 不同，整体可复现
    history = {"epoch": [], "train_loss": [], "train_acc": [], "test_acc": []}
    n = len(Xtr)

    for ep in range(epochs):
        t0 = time.perf_counter()
        # 每 epoch 重新加噪（数据增强），去噪真实执行
        Xtr_in = make_inputs(Xtr, kernel, use_noise, use_denoise, rng=rng)
        perm = rng.permutation(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            cnn.forward(data.to_nchw(Xtr_in[idx]))
            cnn.backward(ytr[idx])
            for p, g in zip(cnn.params(), cnn.grads()):
                p -= lr * g
        # 指标（train loss 用小批量算，避免大 batch 前向的滑窗中间量爆内存/拖慢）
        loss_sub = min(_LOSS_SUBSET, n)
        tr_loss = cnn.loss(data.to_nchw(Xtr_in[:loss_sub]), ytr[:loss_sub])
        sub = min(_EVAL_SUBSET, n)
        tr_in_sub = Xtr_in[:sub]
        tr_acc = predict_acc(cnn, tr_in_sub, ytr[:sub], batch_size)
        te_sub = min(_EVAL_SUBSET, len(Xte_in))
        te_acc = predict_acc(cnn, Xte_in[:te_sub], yte[:te_sub], batch_size)
        dt = time.perf_counter() - t0
        history["epoch"].append(ep + 1)
        history["train_loss"].append(float(tr_loss))
        history["train_acc"].append(float(tr_acc))
        history["test_acc"].append(float(te_acc))
        log(f"  epoch {ep+1:2d}/{epochs}  loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
            f"test_acc={te_acc:.4f}  ({dt:.1f}s)")

    final_acc = predict_acc(cnn, Xte_in, yte, batch_size)
    log(f"  [final] full test acc = {final_acc:.4f}")

    model_path = _root(os.path.join(config.OUTPUT_DIR, f"model{sfx}.npz"))
    save_model(cnn, model_path)
    curve_path = viz.plot_train_curve(
        history, os.path.join(config.OUTPUT_DIR, f"train_curve{sfx}.png"),
        title=f"train curve [{tag}] noise={use_noise} denoise={use_denoise}")
    cm = confusion(cnn, Xte_in, yte, batch_size)
    cm_path = viz.plot_confusion_matrix(
        cm, os.path.join(config.OUTPUT_DIR, f"confusion_matrix{sfx}.png"),
        title=f"confusion matrix [{tag}] acc={final_acc*100:.2f}%")
    log(f"  saved: {model_path}")
    log(f"  saved: {curve_path}")
    log(f"  saved: {cm_path}")
    return {"history": history, "test_acc": final_acc, "model_path": model_path,
            "confusion": cm, "Xte_in": Xte_in, "yte": yte, "Xte": Xte, "cnn": cnn}


def denoise_showcase(seed=None, m=5, path=None):
    """真实 MNIST 数字去噪三栏对比（含与干净图的 MSE，如实呈现边缘模糊代价）。"""
    seed = config.SEED if seed is None else seed
    _, _, Xte, yte = data.load_mnist(log=(lambda *a: None))
    rng = np.random.default_rng(seed + 1)
    clean = Xte[:m].copy()
    noisy = data.add_gaussian_noise(clean, config.NOISE_SIGMA, rng=rng)
    den = denoise_batch(noisy, gaussian_kernel())
    mse_n = float(np.mean((noisy - clean) ** 2))
    mse_d = float(np.mean((den - clean) ** 2))
    p = viz.plot_denoise_triplet(clean, noisy, den,
                                 path or os.path.join(config.OUTPUT_DIR, "denoise_mnist_compare.png"),
                                 title="real MNIST: clean / noisy / denoised")
    print(f"  denoise showcase: noisy MSE={mse_n:.4f}  denoised MSE={mse_d:.4f}  -> {p}")
    return mse_n, mse_d, p


def main():
    ap = argparse.ArgumentParser(description="MNIST CNN 训练（手写反向，真实去噪管线）")
    ap.add_argument("--n-train", type=int, default=config.N_TRAIN)
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--lr", type=float, default=config.LR)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--no-noise", action="store_true", help="训练/测试输入不加噪（组 A）")
    ap.add_argument("--no-denoise", action="store_true", help="不加去噪（组 A/C）")
    ap.add_argument("--tag", type=str, default="main", help="输出后缀，main 用默认文件名")
    ap.add_argument("--showcase", action="store_true", help="额外生成真实 MNIST 去噪对比图")
    args = ap.parse_args()

    res = run(n_train=args.n_train, epochs=args.epochs, lr=args.lr,
              batch_size=args.batch_size, use_noise=not args.no_noise,
              use_denoise=not args.no_denoise, seed=args.seed, tag=args.tag)
    if args.showcase or args.tag == "main":
        denoise_showcase(seed=args.seed)
    print(f"[done] tag={args.tag} final_test_acc={res['test_acc']*100:.2f}%")


if __name__ == "__main__":
    main()
