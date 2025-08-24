# LADiT-Pipe

## 概要
LADiT-Pipeは、ローカル環境で動作する、高精度な話者分離・文字起こしパイプラインです。NVIDIA GPU (8GB VRAM以上推奨) を活用し、効率的な音声・動画ファイルの処理と、話者ごとの文字起こしを提供します。

## 必須要件
- Python 3.10+
- ffmpeg
- CUDA対応のNVIDIA GPU (8GB VRAM以上推奨)

## セットアップ方法

1.  **リポジトリのクローン:**
    ```bash
    git clone https://github.com/your-repo/ladit-pipe.git
    cd ladit-pipe
    ```
    (注: `https://github.com/your-repo/ladit-pipe.git` は実際のGitHubリポジトリURLに置き換えてください。)

2.  **依存関係のインストール:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Hugging Face `pyannote` モデルの利用規約への同意:**
    `pyannote.audio` ライブラリを使用するには、Hugging Faceのウェブサイトで以下のモデルの利用規約に同意する必要があります。
    - [pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization)
    - [pyannote/embedding](https://huggingface.co/pyannote/embedding)
    Hugging Faceアカウントでログインし、各モデルページにアクセスして「Agree and access repository」ボタンをクリックしてください。同意後、Hugging Faceのアクセストークンを環境変数 `HUGGINGFACE_HUB_TOKEN` に設定するか、スクリプト実行時に `--hf-token` 引数で渡す必要があります。

## 基本的な使い方

`ladit_pipe.main` スクリプトを使用して、音声・動画ファイルの処理を実行します。

### `meeting` プリセットの実行例

会議などの複数話者が存在する音声・動画ファイルに適しています。`--max-speakers` で最大話者数を指定することで、話者分離の精度を向上させることができます。

```bash
python3 -m ladit_pipe.main "path/to/your/input_file.mp4" --preset meeting --output ./output_dir --max-speakers 7 --hf-token YOUR_HUGGING_FACE_TOKEN
```

-   `"path/to/your/input_file.mp4"`: 処理したい音声または動画ファイルへのパス。ディレクトリを指定することも可能です。
-   `--preset meeting`: `meeting` プリセットを使用することを指定します。
-   `--output ./output_dir`: 結果を出力するディレクトリを指定します。
-   `--max-speakers 7`: 検出する最大話者数を7人に設定します。実際の話者数に合わせて調整してください。
-   `--hf-token YOUR_HUGGING_FACE_TOKEN`: Hugging Faceのアクセストークンを指定します。

### `walking` プリセットの実行例

一人話者の音声ファイルや、比較的短い音声ファイルに適しています。

```bash
python3 -m ladit_pipe.main "path/to/your/audio.wav" --preset walking --output ./output_dir --hf-token YOUR_HUGGING_FACE_TOKEN
```

### `--verbose` フラグ

詳細なログ出力を有効にするには、`--verbose` フラグを追加してください。デバッグや問題の特定に役立ちます。

```bash
python3 -m ladit_pipe.main "path/to/your/input_file.mp4" --preset meeting --output ./output_dir --max-speakers 7 --hf-token YOUR_HUGGING_FACE_TOKEN --verbose
```
