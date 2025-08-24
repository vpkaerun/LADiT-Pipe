from pathlib import Path
from typing import List, Set

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
