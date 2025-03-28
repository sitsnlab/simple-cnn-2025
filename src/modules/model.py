import torch
import torch.nn as nn


class MyCNN(nn.Module):
    """
    畳み込みニューラルネットワーク (CNN) の実装。

    2 層の畳み込み層と 2 層の全結合層 (2層目は出力) から構成されるシンプルな CNN。
    ReLU 活性化関数と最大プーリング層を使用し、最終的に分類を行う。

    Attributes:
        relu (nn.ReLU): 活性化関数 ReLU。
        pool (nn.MaxPool2d): 2×2 の最大プーリング層。
        conv1 (nn.Conv2d): 3 チャネル入力、16 チャネル出力の畳み込み層。
        conv2 (nn.Conv2d): 16 チャネル入力、32 チャネル出力の畳み込み層。
        fc1 (nn.Linear): 全結合層、入力次元は `16 * 16 * 32`、出力次元は 256。
        fc2 (nn.Linear): 全結合層、入力次元は 256、出力次元は `num_classes`。
    """

    def __init__(self, num_classes: int):
        """
        モデルのレイヤーを定義し、重みを初期化する。

        Args:
            num_classes (int): 出力層のクラス数（分類タスクのクラス数）。
        """
        super(MyCNN, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 畳み込み層
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)

        # 全結合層
        self.fc1 = nn.Linear(32 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)

        # 重みの初期化
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播を実行し、入力データを分類する。

        Args:
            x (torch.Tensor): 入力画像のバッチ (shape: [batch_size, 3, 256, 256])。

        Returns:
            torch.Tensor: クラスごとのスコア (shape: [batch_size, num_classes])。
        """
        x = self.conv1(x)  # [3, 256, 256] -> [16, 128, 128]
        x = self.relu(x)
        x = self.pool(x)  # [16, 128, 128] -> [16, 64, 64]

        x = self.conv2(x)  # [16, 64, 64] -> [32, 32, 32]
        x = self.relu(x)
        x = self.pool(x)  # [32, 32, 32] -> [32, 16, 16]

        # 平坦化処理（全結合層に入力するため）
        x = x.view(x.size(0), -1)  # [32, 16, 16] -> [8,192 (32*16*16)]

        x = self.fc1(x)  # [8,192 (32*16*16)] -> [256]
        x = self.relu(x)
        x = self.fc2(x)  # [256] -> [num_classes]

        return x

    def _initialize_weights(self) -> None:
        """
        モデルの重みを適切に初期化する。

        畳み込み層および全結合層の重みを He (Kaiming) 初期化し、
        バイアスを一様分布でランダムに設定する。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.uniform_(m.bias, a=-0.1, b=0.1)
