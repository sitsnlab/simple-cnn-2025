from datetime import datetime

import torch
import torchvision.transforms as transforms

# デバイス設定
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ハイパーパラメータ
SEED = 1
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 10
IMAGE_SIZE = (256, 256)

# 設定の表示
IS_DISPLAY_CONFIG = True

# データセットのディレクトリ
# 分類に使う画像のディレクトリを指定
IMAGE_DIRS = {
    "train": [
        "../data/small-places/train/wave",
        "../data/small-places/train/tower",
        "../data/small-places/train/escalotor",
        # "../data/small-places/train/wind_farm",
        # "../data/small-places/train/airfield",
    ],
    "test": [
        "../data/small-places/test/wave",
        "../data/small-places/test/tower",
        "../data/small-places/test/escalotor",
        # "../data/small-places/test/wind_farm",
        # "../data/small-places/test/airfield",
    ],
}

# 画像の変換処理
TRANSFORM = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ]
)


BASE_RESULT_DIR = "../results/"

# リザルトフォルダのパス(パラメータ＋日時)
RESULT_DIR = (
    f"batch{BATCH_SIZE}_lr{LR}_epoch{EPOCHS}_imgsize{IMAGE_SIZE[0]}_"
    + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    + "/"
)


def display_config() -> None:
    if IS_DISPLAY_CONFIG:
        print("\n" + "===================== CONFIG =====================")
        print("DEVICE:", DEVICE)
        print("SEED:", SEED)
        print("BATCH_SIZE:", BATCH_SIZE)
        print("LR:", LR)
        print("EPOCHS:", EPOCHS)
        print("IMAGE_SIZE:", IMAGE_SIZE)
        print("OUTPUT_CLASSES:", len(IMAGE_DIRS["train"]))
        print("==================================================" + "\n")
