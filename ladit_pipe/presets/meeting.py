# ladit_pipe/presets/meeting.py (Final Version)

import logging
import argparse
import tempfile
import shutil
import gc
import subprocess
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import whisper
from tqdm import tqdm
from pyannote.audio import Pipeline
from pyannote.core import Annotation

# LADiT-Pipeの内部モジュールをインポート
from ladit_pipe.utils.ffmpeg_wrapper import (
    split_into_chunks_with_progress,
)
from ladit_pipe.core.diarization import (
    perform_global_diarization,
)
from ladit_pipe.core.transcription import (
    create_transcription_chunks,
    transcribe_chunk,
)
from ladit_pipe.core.export import (
    merge_results,
)

# ロガー設定
logger = logging.getLogger(__name__)

# 新しい定数：メモリ管理可能な中間チャンクの長さ（秒）
PROCESSING_CHUNK_DURATION_SEC = 600  # 10分

import subprocess

def get_duration_sec(file_path: Path) -> float:
    """ffprobeを使ってファイルの再生時間を取得する"""
    try:
        result = subprocess.check_output([
            'ffprobe', '-v', 'error', '-show_entries',
            'format=duration', '-of',
            'default=noprint_wrappers=1:nokey=1', str(file_path)
        ]).decode('utf-8').strip()
        return float(result)
    except Exception as e:
        logger.error(f"Duration取得エラー for {file_path}: {e}")
        return 0.0

def convert_to_wav_with_progress(input_file: Path, temp_dir: Path, output_file: Optional[Path] = None, sample_rate: int = 16000) -> Path:
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
        "-i", str(input_file),
        "-ar", str(sample_rate),
        "-ac", "1",
        "-c:a", "pcm_s16le",
        "-progress", "pipe:1",
        str(output_file)
    ]

    progress_re = re.compile(r"out_time_ms=(\d+)")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    pbar = tqdm(total=duration, unit="sec", desc=f"Converting {input_file.name}", leave=False)
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


def run(args: argparse.Namespace):
    """
    ミーティングプリセット（最終ハイブリッド版）を実行します。
    巨大ファイルをメモリ管理可能なチャンクに分割し、それぞれを処理することで、
    ファイルの長さに依存しない、安定した高精度処理を実現します。
    """
    logger.info("Meeting preset [Phoenix Architecture] processing started.")
    temp_dir = Path(tempfile.mkdtemp(prefix="ladit_pipe_phoenix_"))
    logger.debug(f"一時ディレクトリを作成しました: {temp_dir}")

    try:
        input_file = args.input
        logger.info(f"入力ファイル {input_file.name} を処理します。")

        # --- ステップ 1: WAV変換 ---
        wav_file = convert_to_wav_with_progress(input_file, temp_dir)
        if not wav_file: return

        # --- ステップ 2: グローバル話者分離 (並列で実行可能) ---
        logger.info("ファイル全体の話者分離をバックグラウンドで（概念的に）開始します...")
        global_diarization = perform_global_diarization(
            wav_file=wav_file, temp_dir=temp_dir, hf_token=args.hf_token,
            device=args.device, min_speakers=args.min_speakers, max_speakers=args.max_speakers
        )
        logger.info(f"グローバル話者分離が完了。{len(global_diarization.labels())}人の話者を検出。")

        # --- ステップ 3: 巨大WAVを処理可能な中間チャンクに分割 ---
        logger.info(f"{PROCESSING_CHUNK_DURATION_SEC}秒単位の処理用チャンクに分割します...")
        processing_chunks = split_into_chunks_with_progress(
            wav_file, temp_dir, PROCESSING_CHUNK_DURATION_SEC
        )

        # --- ステップ 4: 中間チャンクごとの文字起こし (最終修正版) ---
        all_transcription_results = []
        whisper_model = whisper.load_model(args.whisper_model, device=args.device)

        for i, chunk_path in enumerate(tqdm(processing_chunks, desc="Transcribing Chunks")):
            chunk_offset_sec = i * PROCESSING_CHUNK_DURATION_SEC
            logger.debug(f"Transcribing chunk {i+1}/{len(processing_chunks)} (Offset: {chunk_offset_sec}s)")

            transcription_result_dict = transcribe_chunk(chunk_path, whisper_model)
            
            if transcription_result_dict and "segments" in transcription_result_dict:
                for segment in transcription_result_dict["segments"]:
                    global_start = chunk_offset_sec + segment["start"]
                    global_end = chunk_offset_sec + segment["end"]
                    all_transcription_results.append({
                        "start": global_start,
                        "end": global_end,
                        "text": segment["text"],
                    })

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info("全てのチャンクの文字起こしが完了しました。")
        del whisper_model

        # --- ステップ 5: 最終結果のマージとエクスポート ---
        logger.info("最終結果をマージしてファイルに出力します...")
        session_name = input_file.stem
        output_files = merge_results(
            transcription_results=all_transcription_results,
            global_diarization=global_diarization,
            output_dir=args.output,
            session_name=session_name,
        )

        for format_type, file_path in output_files.items():
            logger.info(f"  -> {format_type.upper()} ファイルを出力しました: {file_path}")

    except Exception as e:
        logger.error(f"処理中に致命的なエラー: {e}", exc_info=True)

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logger.debug(f"一時ディレクトリ {temp_dir} を削除しました。")

    logger.info("Phoenix Architecture processing finished.")