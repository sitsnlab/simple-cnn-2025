import csv
import os
import random
import shutil

import torch

from config import BASE_RESULT_DIR, RESULT_DIR, SEED


def set_seed() -> None:
    """
    乱数シードを設定し、再現性を確保する。

    PyTorch および Python の乱数シードを固定し、CUDA を使用する際の再現性も確保するための設定を行う。
    """
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # cuDNN の再現性を確保するための設定
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_result_directory() -> str:
    """
    結果を保存するためのディレクトリを作成する。

    `BASE_RESULT_DIR` と `RESULT_DIR` を結合し、存在しない場合はディレクトリを作成する。

    Returns:
        str: 作成された結果保存用ディレクトリのパス。
    """
    result_dir = os.path.join(BASE_RESULT_DIR, RESULT_DIR)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def save_config(result_dir: str) -> None:
    """
    設定ファイルを指定したディレクトリにコピーする。

    `config.py` ファイルを結果保存用ディレクトリ (`result_dir`) にコピーし、
    実験設定を後から再現可能にする。

    Args:
        result_dir (str): 設定ファイルを保存するディレクトリのパス。
    """
    shutil.copy("./config.py", result_dir)


def save_results_to_csv(result_dir: str, csv_columns: list) -> str:
    """
    結果を保存するための CSV ファイルを初期化する。

    指定したカラム名でヘッダーを作成し、CSV ファイルを `result_dir` に保存する。

    Args:
        result_dir (str): CSV ファイルを保存するディレクトリのパス。
        csv_columns (list): CSV のヘッダーとなるカラム名のリスト。

    Returns:
        str: 作成された CSV ファイルのパス。
    """
    csv_file = os.path.join(result_dir, "result.csv")
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(csv_columns)
    return csv_file


def save_epoch_results(
    csv_file: str,
    epoch: int,
    running_loss: float,
    train_loader,
    train_accuracy: float,
    test_accuracy: float,
    start_time: float,
    end_time: float,
) -> None:
    """
    エポックごとの学習結果を CSV ファイルに書き込む。

    学習の進行状況を記録し、各エポックごとの損失、精度、経過時間を CSV に追加する。

    Args:
        csv_file (str): 結果を保存する CSV ファイルのパス。
        epoch (int): 現在のエポック数。
        running_loss (float): 学習時の累積損失値。
        train_loader: 訓練データの DataLoader（バッチ数の計算に使用）。
        train_accuracy (float): 訓練データでの精度。
        test_accuracy (float): テストデータでの精度。
        start_time (float): エポック開始時の時間（time.time() の値）。
        end_time (float): エポック終了時の時間（time.time() の値）。
    """
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        time_taken = end_time - start_time
        writer.writerow(
            [
                epoch + 1,
                running_loss / len(train_loader),
                train_accuracy,
                test_accuracy,
                time_taken,
            ]
        )


def save_best_model(
    model: torch.nn.Module, loss: float, best_loss: float, result_dir: str
) -> float:
    """
    最良のモデルを保存する。

    現在の損失がこれまでの最小損失よりも小さい場合、モデルの重みを `result_dir` に保存する。

    Args:
        model (torch.nn.Module): 保存する PyTorch モデル。
        loss (float): 現在のエポックの損失値。
        best_loss (float): これまでの最小損失値。
        result_dir (str): モデルを保存するディレクトリのパス。

    Returns:
        float: 更新後の最小損失値。
    """
    if loss < best_loss:
        best_loss = loss
        model_save_path = os.path.join(result_dir, "best_model.pth")
        torch.save(model.state_dict(), model_save_path)
    return best_loss
