import subprocess
import logging
from pathlib import Path
from typing import List, Optional
import re
from tqdm import tqdm

# ロガー設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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
        subprocess.CalledProcessError:
            ffprobeコマンドが失敗した場合。
        ValueError:
            再生時間のパースに失敗した場合。
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
        result = subprocess.run(
            command, check=True, capture_output=True, text=True
        )
        duration_str = result.stdout.strip()
        duration = float(duration_str)
        logger.info(f"Duration for {file_path.name}: {duration:.2f} seconds.")
        return duration
    except subprocess.CalledProcessError as e:
        logger.error(f"ffprobe failed for {file_path.name}: {e.stderr}")
        raise
    except ValueError:
        logger.error(
            f"Failed to parse duration from ffprobe output: {duration_str}"
        )
        raise ValueError(f"Could not parse duration for {file_path.name}")


def split_into_chunks(
    input_file: Path, output_dir: Path, chunk_duration_sec: int
) -> List[Path]:
    """
    ffmpegを使用してファイルを指定した時間でチャンク分割します。
    分割の進捗をプログレスバーで表示します。

    Args:
        input_file (Path): 分割する入力ファイルのパス。
        output_dir (Path): 分割されたチャンクファイルを保存するディレクトリのパス。
        chunk_duration_sec (int): 各チャンクの長さ（秒）。

    Returns:
        List[Path]: 分割後のチャンクファイルのパスのリスト。

    Raises:
        subprocess.CalledProcessError:
            ffmpegコマンドが失敗した場合。
    """
    logger.info(
        f"Splitting {input_file.name} into {chunk_duration_sec}s chunks..."
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # まずffprobeでファイルの総再生時間を取得
    try:
        total_duration_str = (
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(input_file),
                ]
            )
            .decode("utf-8")
            .strip()
        )
        total_duration_sec = float(total_duration_str)
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.warning(
            f"ファイルの再生時間取得に失敗: {e}。"
            "プログレスバーは表示されません。"
        )
        total_duration_sec = None

    # 出力ファイル名のパターン (例: input_file_000.wav, input_file_001.wav)
    output_pattern = output_dir / f"{input_file.stem}_%03d.wav"

    cmd = [
        "ffmpeg",
        "-i",
        str(input_file),
        "-f",
        "segment",
        "-segment_time",
        str(chunk_duration_sec),
        "-c",
        "copy",  # 再エンコードなしで高速に分割
        "-y",  # 既存ファイルを上書き
        "-hide_banner",  # 余計なバナー情報を非表示
        "-loglevel",
        "error",  # 通常のエラー以外のログを抑制
        "-progress",
        "pipe:1",  # 進捗をパイプに出力
        str(output_pattern),
    ]

    # tqdmプログレスバーの設定
    progress_bar = tqdm(
        total=int(total_duration_sec) if total_duration_sec else None,
        desc=f"Splitting {input_file.name}",
        unit="s",
    )

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # stdoutとstderrを統合
        universal_newlines=True,
        encoding="utf-8",
    )

    for line in process.stdout:
        if total_duration_sec:
            # 'out_time_ms' または 'out_time' を探す
            match = re.search(r"out_time_ms=(\d+)", line)
            if not match:
                match = re.search(r"out_time=(.+)", line)

            if match:
                if ":" in match.group(1):  # HH:MM:SS.ms 形式
                    h, m, s = map(float, match.group(1).split(":"))
                    processed_sec = h * 3600 + m * 60 + s
                else:  # マイクロ秒形式
                    processed_sec = int(match.group(1)) / 1_000_000

                # プログレスバーを更新
                progress_bar.update(int(processed_sec) - progress_bar.n)

    progress_bar.close()
    process.wait()

    if process.returncode != 0:
        logger.error(f"ffmpegチャンク分割エラー ({input_file.name})")
        raise subprocess.CalledProcessError(process.returncode, cmd)

    logger.info(
        f"Successfully split {input_file.name} into chunks "
        f"in {output_dir.name}"
    )

    # 生成されたチャンクファイルのパスを収集
    chunk_files = sorted(list(output_dir.glob(f"{input_file.stem}_*.wav")))
    return chunk_files


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
        subprocess.CalledProcessError:
            ffmpegコマンドが失敗した場合。
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
        logger.error(
            f"ffmpeg clip extraction failed for {input_file.name}: {e.stderr}"
        )
        raise


# 【プログレスバー対応版】
def convert_to_wav(
    input_file: Path, temp_dir: Path
) -> Path:
    """
    任意の音声・動画ファイルを16kHzモノラルのWAVファイルに変換する。
    変換の進捗をプログレスバーで表示する。
    """
    output_file = temp_dir / f"{input_file.stem}_converted.wav"

    if output_file.exists():
        logger.info(
            f"変換済みファイルが既に存在するためスキップ: {output_file.name}"
        )
        return output_file

    try:
        # まずffprobeでファイルの総再生時間を取得
        total_duration_str = (
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(input_file),
                ]
            )
            .decode("utf-8")
            .strip()
        )
        total_duration_sec = float(total_duration_str)
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.warning(
            f"ファイルの再生時間取得に失敗: {e}。"
            "プログレスバーは表示されません。"
        )
        total_duration_sec = None

    cmd = [
        "ffmpeg",
        "-i",
        str(input_file),
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        "-y",
        "-hide_banner",  # 余計なバナー情報を非表示
        "-loglevel",
        "error",  # 通常のエラー以外のログを抑制
        "-progress",
        "pipe:1",  # 進捗をパイプに出力
        str(output_file),
    ]

    # tqdmプログレスバーの設定
    progress_bar = tqdm(
        total=int(total_duration_sec) if total_duration_sec else None,
        desc=f"Converting {input_file.name}",
        unit="s",
    )

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # stdoutとstderrを統合
        universal_newlines=True,
        encoding="utf-8",
    )

    for line in process.stdout:
        if total_duration_sec:
            # 'out_time_ms' または 'out_time' を探す
            match = re.search(r"out_time_ms=(\d+)", line)
            if not match:
                match = re.search(r"out_time=(.+)", line)

            if match:
                if ":" in match.group(1):  # HH:MM:SS.ms 形式
                    h, m, s = map(float, match.group(1).split(":"))
                    processed_sec = h * 3600 + m * 60 + s
                else:  # マイクロ秒形式
                    processed_sec = int(match.group(1)) / 1_000_000

                # プログレスバーを更新
                progress_bar.update(int(processed_sec) - progress_bar.n)

    progress_bar.close()
    process.wait()

    if process.returncode != 0:
        logger.error(f"ffmpeg変換エラー ({input_file.name})")
        # process.stderr.read() は使えないので、エラーハンドリングは簡略化
        raise subprocess.CalledProcessError(process.returncode, cmd)

    logger.info(f"変換完了: {input_file.name} -> {output_file.name}")
    return output_file


def convert_to_wav_with_progress(
    input_file: Path,
    temp_dir: Path,
    output_file: Optional[Path] = None,
    sample_rate: int = 16000,
) -> Path:
    """
    ffmpegで音声ファイルを16kHzモノラルWAVに変換し、進捗をtqdmで表示する。
    """
    if output_file is None:
        output_file = temp_dir / f"{input_file.stem}_converted.wav"
    if output_file.exists():
        return output_file

    duration = get_duration_sec(input_file)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_file),
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        "-progress",
        "pipe:1",
        str(output_file),
    ]

    progress_re = re.compile(r"out_time_ms=(\d+)")
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    pbar = tqdm(
        total=duration,
        unit="sec",
        desc=f"Converting {input_file.name}",
        leave=False,
    )
    last_sec = 0
    try:
        for line in p.stdout:
            match = progress_re.search(line)
            if match:
                out_time_ms = int(match.group(1))
                out_time_sec = out_time_ms / 1_000_000
                if out_time_sec > last_sec:
                    pbar.update(out_time_sec - last_sec)
                    last_sec = out_time_sec
        p.wait()
        pbar.n = duration
        pbar.refresh()
        pbar.close()
        if p.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {input_file}")
    finally:
        pbar.close()
    return output_file


def concatenate_files(input_files: List[Path], output_file: Path) -> Path:
    """
    複数のオーディオファイルを連結し、単一のWAVファイルとして出力します。

    Args:
        input_files (List[Path]): 連結する入力ファイルのパスのリスト。
        output_file (Path): 連結されたWAVファイルの出力パス。

    Returns:
        Path: 連結されたWAVファイルのパス。

    Raises:
        subprocess.CalledProcessError:
            ffmpegコマンドが失敗した場合。
        ValueError: 入力ファイルが指定されていない場合。
    """
    if not input_files:
        raise ValueError("連結する入力ファイルが指定されていません。")

    logger.info(f"{len(input_files)}個のファイルを連結します。ファイルのサイズによっては、時間がかかる場合があります...")

    # ffmpegのconcatデムーサーを使用するためのファイルリストを作成
    resolved_paths = []
    for f in input_files:
        resolved_path = f.resolve()
        logger.debug(f"Resolved path for {f.name}: {resolved_path}")
        resolved_paths.append(resolved_path)

    list_file_content = "\n".join([f"file '{p}'" for p in resolved_paths])
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
        logger.error(f"Error during file concatenation: {e}\n")
        raise


def split_into_chunks_with_progress(
    input_file: Path, temp_dir: Path, chunk_duration_sec: int
) -> List[Path]:
    """
    ffmpegを使用してファイルを指定した時間でチャンク分割し、進捗を表示します。

    Args:
        input_file (Path): 分割する入力ファイルのパス。
        temp_dir (Path): 分割されたチャンクを保存する親ディレクトリ。
        chunk_duration_sec (int): 各チャンクの長さ（秒）。

    Returns:
        List[Path]: 分割後のチャンクファイルのパスのリスト。

    Raises:
        subprocess.CalledProcessError:
            ffmpegコマンドが失敗した場合。
    """
    logger.info(
        f"Splitting {input_file.name} into {chunk_duration_sec}s chunks "
        "with progress..."
    )
    output_dir = temp_dir / f"{input_file.stem}_chunks"
    output_dir.mkdir(exist_ok=True)
    output_pattern = output_dir / "chunk_%03d.wav"

    try:
        # まずffprobeでファイルの総再生時間を取得
        total_duration_str = (
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(input_file),
                ]
            )
            .decode("utf-8")
            .strip()
        )
        total_duration_sec = float(total_duration_str)
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.warning(
            f"ファイルの再生時間取得に失敗: {e}。"
            "プログレスバーは表示されません。"
        )
        total_duration_sec = None

    cmd = [
        "ffmpeg",
        "-i",
        str(input_file),
        "-f",
        "segment",
        "-segment_time",
        str(chunk_duration_sec),
        "-c",
        "copy",  # 音声を再エンコードせず、そのままコピーする（高速）
        "-reset_timestamps",
        "1",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-progress",
        "pipe:1",
        str(output_pattern),
    ]

    # tqdmプログレスバーの設定
    progress_bar = tqdm(
        total=int(total_duration_sec) if total_duration_sec else None,
        desc=f"Splitting {input_file.name}",
        unit="s",
    )

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # stdoutとstderrを統合
        universal_newlines=True,
        encoding="utf-8",
    )

    for line in process.stdout:
        if total_duration_sec:
            match = re.search(r"out_time_ms=(\d+)", line)
            if not match:
                match = re.search(r"out_time=(.+)", line)

            if match:
                if ":" in match.group(1):
                    h, m, s = map(float, match.group(1).split(":"))
                    processed_sec = h * 3600 + m * 60 + s
                else:
                    processed_sec = int(match.group(1)) / 1_000_000

                progress_bar.update(int(processed_sec) - progress_bar.n)

    progress_bar.close()
    process.wait()

    if process.returncode != 0:
        logger.error(f"ffmpegチャンク分割エラー ({input_file.name})")
        raise subprocess.CalledProcessError(process.returncode, cmd)

    logger.info(
        f"Successfully split {input_file.name} into chunks "
        f"in {output_dir.name}"
    )

    chunk_files = sorted(list(output_dir.glob("*.wav")))
    return chunk_files