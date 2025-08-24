import subprocess
import logging
from pathlib import Path
from typing import List

# ロガー設定
logger = logging.getLogger(__name__)

# 定数
TARGET_SAMPLE_RATE = 16000


def get_duration_sec(file_path: Path) -> float:
    """
    ffprobeを使用して動画/音声ファイルの再生時間を秒単位で取得します。

    Args:
        file_path (Path): 再生時間を取得するファイルのパス。

    Returns:
        float: ファイルの再生時間（秒）。

    Raises:
        subprocess.CalledProcessError: ffprobeコマンドが失敗した場合。
        ValueError: 再生時間のパースに失敗した場合。
    """
    logger.info(f"Getting duration for {file_path.name}...")
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(file_path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        duration_str = result.stdout.strip()
        duration = float(duration_str)
        logger.info(f"Duration for {file_path.name}: {duration:.2f} seconds.")
        return duration
    except subprocess.CalledProcessError as e:
        logger.error(f"ffprobe failed for {file_path.name}: {e.stderr}")
        raise
    except ValueError:
        logger.error(f"Failed to parse duration from ffprobe output: {duration_str}")
        raise ValueError(f"Could not parse duration for {file_path.name}")


def split_into_chunks(
    input_file: Path, output_dir: Path, chunk_duration_sec: int
) -> List[Path]:
    """
    ffmpegを使用してファイルを指定した時間でチャンク分割します。

    Args:
        input_file (Path): 分割する入力ファイルのパス。
        output_dir (Path): 分割されたチャンクファイルを保存するディレクトリのパス。
        chunk_duration_sec (int): 各チャンクの長さ（秒）。

    Returns:
        List[Path]: 分割後のチャンクファイルのパスのリスト。

    Raises:
        subprocess.CalledProcessError: ffmpegコマンドが失敗した場合。
    """
    logger.info(f"Splitting {input_file.name} into {chunk_duration_sec}s chunks...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 出力ファイル名のパターン (例: input_file_000.wav, input_file_001.wav)
    output_pattern = output_dir / f"{input_file.stem}_%03d.wav"

    command = [
        "ffmpeg",
        "-i",
        str(input_file),
        "-f",
        "segment",
        "-segment_time",
        str(chunk_duration_sec),
        "-c",
        "copy",  # 再エンコードなしで高速に分割
        str(output_pattern),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info(
            f"Successfully split {input_file.name} into chunks in {output_dir.name}"
        )

        # 生成されたチャンクファイルのパスを収集
        chunk_files = sorted(list(output_dir.glob(f"{input_file.stem}_*.wav")))
        return chunk_files
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg chunk splitting failed for {input_file.name}: {e.stderr}")
        raise


def extract_clip(
    input_file: Path, output_file: Path, start_sec: float, end_sec: float
) -> Path:
    """
    ffmpegを使用して、指定された時間範囲で音声クリップを切り出します。

    Args:
        input_file (Path): 切り出す元の入力ファイルのパス。
        output_file (Path): 切り出された音声クリップの出力パス。
        start_sec (float): 切り出し開始時間（秒）。
        end_sec (float): 切り出し終了時間（秒）。

    Returns:
        Path: 切り出された音声クリップのパス。

    Raises:
        subprocess.CalledProcessError: ffmpegコマンドが失敗した場合。
    """
    logger.info(
        f"Extracting clip from {input_file.name} ({start_sec:.2f}s - {end_sec:.2f}s) to {output_file.name}..."
    )
    command = [
        "ffmpeg",
        "-ss",
        str(start_sec),
        "-to",
        str(end_sec),
        "-i",
        str(input_file),
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(output_file),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info(f"Successfully extracted clip to {output_file.name}")
        return output_file
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg clip extraction failed for {input_file.name}: {e.stderr}")
        raise


def convert_to_wav(input_file: Path, output_file: Path) -> Path:
    """
    指定されたオーディオ/ビデオファイルをWAV形式に変換します。

    Args:
        input_file (Path): 変換する入力ファイルのパス。
        output_file (Path): 出力WAVファイルのパス。

    Returns:
        Path: 変換されたWAVファイルのパス。

    Raises:
        subprocess.CalledProcessError: ffmpegコマンドが失敗した場合。
    """
    logger.info(f"Converting {input_file.name} to WAV...")
    command = [
        "ffmpeg",
        "-i",
        str(input_file),
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(output_file),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info(f"Successfully converted {input_file.name} to {output_file.name}")
        return output_file
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg conversion failed for {input_file.name}: {e.stderr}")
        raise


def concatenate_files(input_files: List[Path], output_file: Path) -> Path:
    """
    複数のオーディオファイルを連結し、単一のWAVファイルとして出力します。

    Args:
        input_files (List[Path]): 連結する入力ファイルのパスのリスト。
        output_file (Path): 連結されたWAVファイルの出力パス。

    Returns:
        Path: 連結されたWAVファイルのパス。

    Raises:
        subprocess.CalledProcessError: ffmpegコマンドが失敗した場合。
        ValueError: 入力ファイルが指定されていない場合。
    """
    if not input_files:
        raise ValueError("連結する入力ファイルが指定されていません。")

    logger.info(f"Concatenating {len(input_files)} files to {output_file.name}...")

    # ffmpegのconcatデムーサーを使用するためのファイルリストを作成
    list_file_content = "\n".join([f"file ‘{f.resolve()}’" for f in input_files])
    list_file_path = output_file.parent / "concat_list.txt"
    list_file_path.write_text(list_file_content, encoding="utf-8")

    command = [
        "ffmpeg",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file_path),
        "-c",
        "copy",
        str(output_file),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info(f"Successfully concatenated files to {output_file.name}")
        list_file_path.unlink()  # 一時ファイル削除
        return output_file
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg concatenation failed: {e.stderr}")
        raise
    except Exception as e:
        logger.error(f"Error during file concatenation: {e}")
        raise
