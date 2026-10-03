"""CLI 评测（Phase 5）：从权重文件复现测试集准确率 + 混淆矩阵。

测试输入必须与训练时一致：同样的 加噪/去噪 开关、同样的 seed 派生的固定噪声
（否则噪声不同会导致准确率对不上）。

用法：
    python src/evaluate.py --model outputs/model.npz
    python src/evaluate.py --model outputs/model_A.npz --no-noise --no-denoise
    python src/evaluate.py --model outputs/model_C.npz --no-denoise
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import config
import data
import viz
from cnn import CNN
from filters import gaussian_kernel
from train import load_model, make_inputs, predict_acc, confusion


def main():
    ap = argparse.ArgumentParser(description="MNIST CNN 评测（复现准确率 + 混淆矩阵）")
    ap.add_argument("--model", type=str, default=config.MODEL_PATH)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--no-denoise", action="store_true")
    ap.add_argument("--n-test", type=int, default=10000)
    ap.add_argument("--cm-out", type=str, default=None, help="混淆矩阵输出路径（默认按模型名派生）")
    args = ap.parse_args()

    use_noise = not args.no_noise
    use_denoise = not args.no_denoise

    _, _, Xte, yte = data.load_mnist(log=(lambda *a: None))
    Xte, yte = Xte[:args.n_test], yte[:args.n_test]
    Xte_in = make_inputs(Xte, gaussian_kernel(), use_noise, use_denoise,
                         rng=np.random.default_rng(args.seed + 1))

    cnn = CNN(seed=args.seed)
    load_model(cnn, args.model)
    acc = predict_acc(cnn, Xte_in, yte, args.batch_size)
    print(f"[evaluate] model={args.model} noise={use_noise} denoise={use_denoise} "
          f"n_test={len(yte)}")
    print(f"[evaluate] test accuracy = {acc:.4f} ({acc*100:.2f}%)")

    cm = confusion(cnn, Xte_in, yte, args.batch_size)
    cm_path = args.cm_out or os.path.join(config.OUTPUT_DIR, "confusion_matrix.png")
    p = viz.plot_confusion_matrix(cm, cm_path,
                                  title=f"confusion matrix acc={acc*100:.2f}%")
    print(f"[evaluate] confusion matrix -> {p}")
    return acc


if __name__ == "__main__":
    main()
