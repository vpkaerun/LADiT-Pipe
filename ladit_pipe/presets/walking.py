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
    concatenate_files,
    convert_to_wav_with_progress,
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
from ladit_pipe.utils.file_handler import (
    find_audio_files,
    group_files_by_date,
)

# ロガー設定
logger = logging.getLogger(__name__)

# 新しい定数：メモリ管理可能な中間チャンクの長さ（秒）
PROCESSING_CHUNK_DURATION_SEC = 600  # 10分


def run(args: argparse.Namespace):
    """
    ウォーキングプリセット（最終ハイブリッド版）を実行します。
    入力ファイルを日付ごとにグループ化し、各日付のファイルを連結して、
    Meeting presetと同様の動的な話者分離と文字起こしを行います。
    """
    logger.info("Walking preset [Phoenix Architecture] processing started.")
    # 全体の一時ディレクトリは最後にまとめて削除
    global_temp_dir = Path(tempfile.mkdtemp(prefix="ladit_pipe_walking_global_"))
    logger.debug(f"グローバル一時ディレクトリを作成しました: {global_temp_dir}")

    try:
        input_path = args.input
        logger.info(f"入力パス {input_path.name} を処理します。")

        # --- ステップ 1: 音声ファイルの検索と日付によるグループ化 ---
        all_audio_files = find_audio_files(input_path)
        if not all_audio_files:
            logger.warning(f"指定されたパス {input_path} に音声ファイルが見つかりませんでした。")
            return

        grouped_files = group_files_by_date(all_audio_files)
        if not grouped_files:
            logger.warning("日付でグループ化できるファイルが見つかりませんでした。")
            return

        logger.info(f"{len(grouped_files)}個のファイルグループを検出しました。")

        # Whisperモデルを一度だけロード
        whisper_model = whisper.load_model(args.whisper_model, device=args.device)
        logger.info(f"Whisperモデル {args.whisper_model} をロードしました。")

        # --- ステップ 2: 各日付グループを処理 ---
        for date_str, files_in_group in grouped_files.items():
            logger.info(f"日付グループ {date_str} の処理を開始します。ファイル数: {len(files_in_group)}")

            # 日付グループごとの一時ディレクトリ
            date_temp_dir = global_temp_dir / date_str
            date_temp_dir.mkdir(exist_ok=True)
            logger.debug(f"日付グループ用一時ディレクトリを作成しました: {date_temp_dir}")

            try:
                # 2.1: グループ内のファイルを連結
                # 連結されたファイル名には日付を含める
                
                # 各ファイルを一時的にWAVに変換してから連結
                converted_files_for_concat = []
                for original_file in files_in_group:
                    converted_file = convert_to_wav_with_progress(original_file, date_temp_dir)
                    if converted_file:
                        converted_files_for_concat.append(converted_file)
                    else:
                        logger.error(f"ファイル {original_file.name} のWAV変換に失敗しました。このファイルをスキップします。")
                        
                if not converted_files_for_concat:
                    logger.error(f"日付グループ {date_str} の連結可能なファイルが見つかりませんでした。このグループの処理をスキップします。")
                    continue

                concatenated_output_path = date_temp_dir / f"concatenated_{date_str}.wav"
                concatenated_wav_file = concatenate_files(converted_files_for_concat, concatenated_output_path)
                logger.info(f"日付 {date_str} のファイルを {concatenated_wav_file.name} に連結しました。")

                # 2.2: 連結されたWAVファイルを処理 (Meeting presetと同様のロジック)
                logger.info(f"連結ファイル {concatenated_wav_file.name} の処理を開始します。")

                # --- ステップ 2.2.1: WAV変換 (既にWAVなのでスキップまたは形式確認) ---
                # concatenate_filesがWAVを返すので、ここでは不要だが、念のためconvert_to_wav_with_progressを呼ぶ
                # convert_to_wav_with_progressは既存ファイルがあればスキップする
                wav_file_for_processing = convert_to_wav_with_progress(concatenated_wav_file, date_temp_dir)
                if not wav_file_for_processing:
                    logger.error(f"WAV変換に失敗しました: {concatenated_wav_file.name}")
                    continue

                # --- ステップ 2.2.2: グローバル話者分離 ---
                logger.info("ファイル全体の話者分離をバックグラウンドで（概念的に）開始します...")
                global_diarization = perform_global_diarization(
                    wav_file=wav_file_for_processing, temp_dir=date_temp_dir, hf_token=args.hf_token,
                    device=args.device, min_speakers=args.min_speakers, max_speakers=args.max_speakers
                )
                logger.info(f"グローバル話者分離が完了。{len(global_diarization.labels())}人の話者を検出。")

                # --- ステップ 2.2.3: 巨大WAVを処理可能な中間チャンクに分割 ---
                logger.info(f"{PROCESSING_CHUNK_DURATION_SEC}秒単位の処理用チャンクに分割します...")
                processing_chunks = split_into_chunks_with_progress(
                    wav_file_for_processing, date_temp_dir, PROCESSING_CHUNK_DURATION_SEC
                )

                # --- ステップ 2.2.4: 中間チャンクごとの文字起こし ---
                all_transcription_results = []
                for i, chunk_path in enumerate(tqdm(processing_chunks, desc=f"Transcribing Chunks for {date_str}")):
                    chunk_offset_sec = i * PROCESSING_CHUNK_DURATION_SEC
                    logger.debug(f"Transcribing chunk {i+1}/{len(processing_chunks)} (Offset: {chunk_offset_sec}s)")

                    transcription_result_dict = transcribe_chunk(chunk_path, whisper_model)

                    if transcription_result_dict and "segments" in transcription_result_dict:
                        for segment in transcription_result_dict["segments"]:
                            all_transcription_results.append({
                                "start": chunk_offset_sec + segment["start"],
                                "end": chunk_offset_sec + segment["end"],
                                "text": segment["text"],
                            })

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                logger.info(f"日付グループ {date_str} の全てのチャンクの文字起こしが完了しました。")

                # --- ステップ 2.2.5: 最終結果のマージとエクスポート ---
                logger.info(f"日付グループ {date_str} の最終結果をマージしてファイルに出力します...")
                session_name = f"walking_log_{date_str}" # セッション名に日付を含める
                output_files = merge_results(
                    transcription_results=all_transcription_results,
                    global_diarization=global_diarization,
                    output_dir=args.output,
                    session_name=session_name,
                )

                # ここでテキストファイルのパスを取得
                txt_file = args.output / f"{session_name}.txt"

                for format_type, file_path in output_files.items():
                    logger.info(f"  -> {format_type.upper()} ファイルを出力しました: {file_path}")

                logger.info(f"日付グループ {date_str} の処理が完了しました。")

            except Exception as e:
                logger.error(f"日付グループ {date_str} の処理中にエラー: {e}", exc_info=True)
                # 特定の日付グループでエラーが発生しても、他のグループの処理は続行する

            finally:
                if date_temp_dir.exists():
                    shutil.rmtree(date_temp_dir)
                    logger.debug(f"日付グループ用一時ディレクトリ {date_temp_dir} を削除しました。")

        del whisper_model # 全てのグループ処理後にモデルをアンロード
        logger.info("Whisperモデルをアンロードしました。")

    except Exception as e:
        logger.error(f"処理中に致命的なエラー: {e}", exc_info=True)

    finally:
        if global_temp_dir.exists():
            shutil.rmtree(global_temp_dir)
            logger.debug(f"グローバル一時ディレクトリ {global_temp_dir} を削除しました。")

    logger.info("Walking preset processing finished.")
