import datetime
import sys

import torch
import torch.nn as nn
import torch.optim as optim

import modules.utils as utils
from config import BATCH_SIZE, DEVICE, EPOCHS, IMAGE_DIRS, LR, TRANSFORM, display_config
from modules.dataset import MyDataset
from modules.model import MyCNN

# 乱数シードの初期化
utils.set_seed()

# 設定情報の表示
display_config()

# 学習のデータローダーの作成
train_dataset = MyDataset(IMAGE_DIRS["train"], TRANSFORM)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)

# テストのデータローダーの取得
test_dataset = MyDataset(IMAGE_DIRS["test"], TRANSFORM)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False
)

# モデルの初期化
num_classes = len(IMAGE_DIRS["train"])
model = MyCNN(num_classes).to(DEVICE)

# 損失関数の設定
criterion = nn.CrossEntropyLoss().to(DEVICE)

# 最適化関数の設定
optimizer = optim.Adam(model.parameters(), lr=LR)

# 結果を保存するディレクトリの作成
result_dir = utils.create_result_directory()

# 設定ファイルの保存
utils.save_config(result_dir)

# 結果を保存するためのCSVファイルの初期化
csv_columns = ["Epoch", "Loss", "Train_Accuracy", "Test_Accuracy", "time"]
csv_file = utils.save_results_to_csv(result_dir, csv_columns)

# 学習ループ開始
print("Training Started!")

# 最小の損失値を保持する変数
min_running_loss = sys.float_info.max

# エポックごとの学習ループ
for epoch in range(EPOCHS):
    running_loss = 0.0  # エポック内の累積損失
    correct = 0  # 正解数カウンター
    total = 0  # サンプル数カウンター

    """ 学習ステップ """

    # 学習モードに変更
    model.train()

    # エポック開始時の時刻を記録
    epoch_start_time = datetime.datetime.now()

    # 学習データをバッチごとに処理
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()  # 勾配の初期化
        outputs = model(images)  # モデルの順伝播
        loss = criterion(outputs, labels)  # 損失の計算
        loss.backward()  # 逆伝播（勾配計算）
        optimizer.step()  # パラメータの更新

        running_loss += loss.item()  # 損失の累積計算

    # エポック終了時の時刻を記録
    epoch_end_time = datetime.datetime.now()

    """ 評価ステップ """

    # 評価モードに変更
    model.eval()

    # 学習データでの精度計算
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total

    # テストデータでの精度計算
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct / total

    # エポックごとの結果を表示
    print(
        f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%"
    )

    # エポックごとの結果をCSVに保存
    utils.save_epoch_results(
        csv_file,
        epoch,
        running_loss,
        train_loader,
        train_accuracy,
        test_accuracy,
        epoch_start_time,
        epoch_end_time,
    )

    # 最良モデルを保存
    min_running_loss = utils.save_best_model(
        model, running_loss, min_running_loss, result_dir
    )


print("Training Finished!")
