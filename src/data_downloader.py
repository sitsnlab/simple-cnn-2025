import os
import zipfile

import gdown

# ダウンロードするフォルダのID
GOOGLE_DRIVE_ID = "1KcZHgB4N7BKZbGd9aFwDaF8QdmcuMeVM"

# ダウンロードするローカルのフォルダ
DOWNLOAD_DIR = "../data/"

# small-placesフォルダが存在しない場合、ZIPファイルをダウンロードして解凍
if not os.path.exists(os.path.join(DOWNLOAD_DIR, "small-places")):
    print("small-placesフォルダが存在しないため、ダウンロードを開始します。")

    # ダウンロードしたZIPファイルのパス
    zip_file_path = os.path.join(DOWNLOAD_DIR, "downloaded_file.zip")

    # ZIPファイルをダウンロード
    print(f"ZIPファイルをダウンロード中... ({zip_file_path})")
    gdown.download(id=GOOGLE_DRIVE_ID, output=zip_file_path, quiet=False)
    print("ダウンロード完了。")

    # ZIPファイルを解凍
    print(f"ZIPファイルを解凍中... {zip_file_path}")
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(DOWNLOAD_DIR)
    print("解凍完了。")

    # ダウンロードしたZIPファイルを削除
    print(f"ダウンロードしたZIPファイルを削除中... {zip_file_path}")
    os.remove(zip_file_path)
    print("削除完了。")
else:
    print("small-placesフォルダはすでに存在します。")
