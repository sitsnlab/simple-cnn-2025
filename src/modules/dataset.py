import os
import random
from typing import Dict, List, Optional, Tuple

import torch
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """
    画像データセットを作成するカスタム Dataset クラス。

    指定されたディレクトリから画像を読み込み、指定された前処理を適用して
    PyTorch の Dataset として使用できる形式にする。

    Attributes:
        transform (torchvision.transforms.Compose): 画像に適用する前処理。
        loader (Callable): 画像を読み込む関数。
        image_paths (List[str]): 画像ファイルのパス一覧。
        labels (List[int]): 画像のラベル一覧。
        class_to_idx (Dict[str, int]): クラス名をインデックスにマッピングする辞書。
    """

    def __init__(
        self,
        image_dirs: List[str],
        transform: torchvision.transforms.Compose,
        num_samples: Optional[int] = None,
    ):
        """
        MyDataset クラスのコンストラクタ。

        Args:
            image_dirs (List[str]): 画像を格納するディレクトリのリスト。
            transform (torchvision.transforms.Compose): 画像に適用する前処理。
            num_samples (Optional[int], optional): 使用する画像の最大数。デフォルトは None（全画像を使用）。
        """
        self.transform = transform
        self.loader = datasets.folder.default_loader
        self.image_paths: List[str] = []
        self.labels: List[int] = []
        self.class_to_idx: Dict[str, int] = {dir: i for i, dir in enumerate(image_dirs)}

        # データセットの画像パスとラベルを取得
        for dir in self.class_to_idx:
            if not os.path.isdir(dir):
                continue
            images = [
                os.path.join(dir, f)
                for f in os.listdir(dir)
                if f.endswith((".png", ".jpg", ".jpeg", ".JPG"))
            ]
            self.image_paths.extend(images)
            self.labels.extend([self.class_to_idx[dir]] * len(images))

        # サンプル数が指定されている場合は、その数だけランダムにサンプル
        if num_samples is not None and num_samples < len(self.image_paths):
            indices = random.sample(range(len(self.image_paths)), num_samples)
            self.image_paths = [self.image_paths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

    def __len__(self) -> int:
        """
        データセットのサイズを返す。

        Returns:
            int: データセット内のサンプル数。
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        指定されたインデックスのデータを取得する。

        Args:
            idx (int): 取得するデータのインデックス。

        Returns:
            Tuple[torch.Tensor, int]: 画像データと対応するラベル。
        """
        label: int = self.labels[idx]
        image: torch.Tensor = self.loader(self.image_paths[idx])
        if self.transform:
            image: torch.Tensor = self.transform(image)
        return image, label
