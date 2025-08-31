# ladit_pipe/presets/meeting.py (Load-Balanced Phoenix Architecture - FINAL)

import logging
import argparse
import tempfile
import shutil
import gc
from pathlib import Path

import torch
import whisper
from tqdm import tqdm

# LADiT-Pipeの内部モジュールをインポート
from ladit_pipe.utils.ffmpeg_wrapper import (
    convert_to_wav_with_progress,
    split_into_chunks_with_progress,
)
from ladit_pipe.core.diarization import perform_chunk_diarization # 新しい関数名
from ladit_pipe.core.transcription import transcribe_chunk_with_segments # 新しい関数名
from ladit_pipe.core.export import merge_and_export_results # 新しい関数名

# ロガー設定
logger = logging.getLogger(__name__)

# 処理チャンクの長さ（秒）
PROCESSING_CHUNK_DURATION_SEC = 600  # 10分


def run(args: argparse.Namespace):
    """
    ミーティングプリセット（最終負荷分散版）を実行します。
    """
    logger.info("Meeting preset [Load-Balanced Phoenix Architecture] processing started.")
    temp_dir = Path(tempfile.mkdtemp(prefix="ladit_pipe_final_"))
    logger.debug(f"一時ディレクトリ: {temp_dir}")

    try:
        input_file = args.input
        logger.info(f"入力ファイル: {input_file.name}")

        # --- ステップ 1: WAV変換 ---
        wav_file = convert_to_wav_with_progress(input_file, temp_dir)
        if not wav_file: return

        # --- ステップ 2: 巨大WAVを処理可能な中間チャンクに分割 ---
        processing_chunks = split_into_chunks_with_progress(
            wav_file, temp_dir, PROCESSING_CHUNK_DURATION_SEC
        )

        # --- ステップ 3: モデルの事前ロード ---
        whisper_model = whisper.load_model(args.whisper_model, device=args.device)
        # Diarizationモデルは、diarizationモジュール内でロードされる

        # --- ステップ 4: チャンクごとの、分離→文字起こし→統合パイプライン ---
        final_timeline = []
        
        for i, chunk_path in enumerate(tqdm(processing_chunks, desc="Processing Chunks")):
            chunk_offset_sec = i * PROCESSING_CHUNK_DURATION_SEC
            logger.info(f"Processing chunk {i+1}/{len(processing_chunks)}...")

            # 4a: この10分チャンクの話者分離
            chunk_diarization = perform_chunk_diarization(
                chunk_path, args.hf_token, args.device, 
                args.min_speakers, args.max_speakers
            )

            # 4b: この10分チャンクの文字起こし (セグメント付き)
            transcription_segments = transcribe_chunk_with_segments(
                chunk_path, whisper_model
            )

            # 4c: この10分チャンクの結果をマージ
            # （このマージロジックは、exportモジュールに新設するのが理想）
            for segment in transcription_segments:
                global_start = chunk_offset_sec + segment["start"]
                global_end = chunk_offset_sec + segment["end"]
                
                # ここで、global_diarizationから話者を特定するロジックが必要
                # （export_claude._find_best_speaker_globalを参考にする）
                speaker = find_speaker_for_segment(chunk_diarization, segment["start"], segment["end"])
                
                final_timeline.append({
                    "start": global_start,
                    "end": global_end,
                    "speaker": speaker,
                    "text": segment["text"],
                })

            # メモリ解放
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # --- ステップ 5: 最終エクスポート ---
        logger.info("最終結果をファイルに出力します...")
        merge_and_export_results(
            final_timeline, args.output, input_file.stem
        )

    except Exception as e:
        logger.error(f"処理中に致命的なエラー: {e}", exc_info=True)

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logger.debug(f"一時ディレクトリを削除しました。")

    logger.info("Load-Balanced Phoenix Architecture processing finished.")


# 新しいヘルパー関数
def find_speaker_for_segment(chunk_diarization, segment_start, segment_end):
    """与えられた時間セグメントに最適な話者を特定します。"""
    # ここに話者特定ロジックを実装します
    # export_claude._find_best_speaker_globalを参考にする
    return "Speaker_TBD"  # ダミーの戻り値
