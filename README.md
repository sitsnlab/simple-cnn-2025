## はじめに
本プロジェクトでは、単純な構造の CNN を実装しています。

## 実行方法
1. `src/config.py` の `IMAGE_DIRS` に分類したいクラスのディレクトリを記述します。
2. 以下のコマンドを実行してください。   
    ```bash
    cd src
    python data_downloader.py # データのダウンロード(初回のみ)
    python main.py # 実行
    ```

## データのリンク
- デモデータ: https://drive.google.com/drive/folders/1IheiqhXHuR5DgX7-DY6Usc6sItROpYbL?usp=drive_link
- ※元データ: https://www.kaggle.com/datasets/benjaminkz/places365
