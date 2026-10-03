"""全项目唯一参数来源（决策 D12）。train / evaluate / app 共用，禁止散落硬编码。

所有相对路径一律以 ROOT（仓库根）为基准解析，禁止硬编码绝对路径。
"""
from pathlib import Path

# 仓库根目录：所有相对路径以此为基准解析，禁止硬编码绝对路径。
ROOT = Path(__file__).resolve().parent.parent

# ---- 数据 / 模型形状 ----
IMG_SIZE = 28
NOISE_SIGMA = 0.3       # 训练时高斯噪声强度
GAUSS_K = 5             # 去噪核大小
GAUSS_SIGMA = 1.2       # 去噪核 sigma
CONV_CHANNELS = (8, 16)
LR = 0.05               # 文档区间 0.01~0.1，取中值起步
BATCH_SIZE = 64
EPOCHS = 10
N_TRAIN = 10000         # 冒烟阶段由命令行覆盖为 500~1000

# ---- 输出 ----
OUTPUT_DIR = "outputs"
MODEL_PATH = "outputs/model.npz"

# ---- 随机种子 ----
SEED = 42

# ---- 数据存放 ----
MNIST_DIR = "data/mnist"   # 决策 D15：data/ 必须进版本库，禁止写进 .gitignore
