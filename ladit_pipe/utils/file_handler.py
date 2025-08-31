from pathlib import Path
from typing import List, Set
import re

# 設定定数
SUPPORTED_AUDIO_FORMATS: Set[str] = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
SUPPORTED_VIDEO_FORMATS: Set[str] = {".mkv", ".mp4", ".mov", ".avi", ".webm"}


def find_audio_files(input_path: Path) -> List[Path]:
    """
    指定されたパス（ファイルまたはディレクトリ）から、サポートされている音声・動画ファイルを検索します。
    ディレクトリが指定された場合、再帰的にファイルを検索します。

    Args:
        input_path (Path): 検索対象のファイルまたはディレクトリのパス。

    Returns:
        List[Path]: 見つかったサポート対象ファイルのパスのリスト。パスはソートされています。
    """
    files = []
    supported_extensions = SUPPORTED_AUDIO_FORMATS | SUPPORTED_VIDEO_FORMATS

    if input_path.is_file():
        if input_path.suffix.lower() in supported_extensions:
            files.append(input_path)
    elif input_path.is_dir():
        for ext in supported_extensions:
            files.extend(input_path.rglob(f"*{ext}"))

    return sorted(files)


def group_files_by_date(files: List[Path]) -> dict[str, List[Path]]:
    """
    ファイル名のプレフィックスから日付を抽出し、同じ日付のファイルをグループ化します。
    ファイル名の形式はYYYY-MM-DDやYYYYMMDDなど、様々な形式に対応します。

    Args:
        files (List[Path]): グループ化するファイルのパスのリスト。

    Returns:
        dict[str, List[Path]]: 日付文字列をキーとし、対応するファイルのリストを値とする辞書。
    """
    grouped_files = {}
    # YYYY-MM-DD または YYYYMMDD 形式の日付を抽出する正規表現
    # ファイル名の先頭から日付パターンを検索
    for file_path in files:
        # YYYY-MM-DD, YYYYMMDD, YYYY_MM_DD などの形式にマッチ
        match = re.search(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})', file_path.stem)
        if match:
            year, month, day = match.groups()
            date_str = f"{year}-{month}-{day}"
            
            if date_str not in grouped_files:
                grouped_files[date_str] = []
            grouped_files[date_str].append(file_path)
        else:
            pass
    return grouped_files
