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
    split_into_chunks_with_progress,
    concatenate_files,
    convert_to_wav_with_progress,
)
from ladit_pipe.core.diarization import (
    perform_chunk_diarization,
)
from ladit_pipe.core.transcription import (
    transcribe_chunk_with_segments,
)
from ladit_pipe.core.export import (
    merge_and_export_results,
    assign_speakers_to_transcription_segments,
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
    ウォーキングプリセット（最終Phoenix版）を実行します。
    """
    logger.info("Walking preset [Phoenix Architecture] processing started.")
    global_temp_dir = Path(tempfile.mkdtemp(prefix="ladit_pipe_walking_phoenix_"))
    logger.debug(f"グローバル一時ディレクトリ: {global_temp_dir}")

    try:
        all_audio_files = find_audio_files(args.input)
        if not all_audio_files:
            logger.warning(f"音声ファイルが見つかりません: {args.input}")
            return

        grouped_files = group_files_by_date(all_audio_files)
        if not grouped_files:
            logger.warning("日付でグループ化できるファイルが見つかりませんでした。")
            return

        whisper_model = whisper.load_model(args.whisper_model, device=args.device)

        for date_str, files_in_group in grouped_files.items():
            logger.info(f"日付グループ {date_str} の処理を開始...")
            date_temp_dir = global_temp_dir / date_str
            date_temp_dir.mkdir(exist_ok=True)

            try:
                # --- ステップ 1: 連結 ---
                converted_wavs = []
                for f in tqdm(files_in_group, desc=f"Converting audio files for {date_str}"):
                    converted_wavs.append(convert_to_wav_with_progress(f, date_temp_dir))
                concatenated_wav_file = date_temp_dir / f"concatenated_{date_str}.wav"
                concatenate_files(converted_wavs, concatenated_wav_file)

                # --- ステップ 2.2: 連結されたWAVファイルを処理 ---
                logger.info(f"連結ファイル {concatenated_wav_file.name} の処理を開始します。")

                # 【！】話者分離は、行わない。
                # その代わり、ファイル全体の長さを持つ、単一話者のダミーAnnotationを作成する。
                from pyannote.core import Annotation, Segment
                from ladit_pipe.utils.ffmpeg_wrapper import get_duration_sec

                duration = get_duration_sec(concatenated_wav_file)
                global_diarization_annotation = Annotation()
                global_diarization_annotation[Segment(0, duration)] = "SPEAKER_00"
                logger.info("単一話者モードとして設定しました。")

                # Annotationオブジェクトを、辞書のリストに変換する
                global_diarization_list = []
                for segment, _, speaker in global_diarization_annotation.itertracks(yield_label=True):
                    global_diarization_list.append({
                        "start": segment.start,
                        "end": segment.end,
                        "speaker": speaker,
                    })

                # --- ステップ 2.3: 中間チャンク分割 (meetingモードと同じ) ---
                processing_chunks = split_into_chunks_with_progress(
                    concatenated_wav_file, date_temp_dir, PROCESSING_CHUNK_DURATION_SEC
                )

                # --- ステップ 2.4: 中間チャンクごとの文字起こし (meetingモードと同じ) ---
                all_transcription_segments_global = []
                for i, chunk_path in enumerate(tqdm(processing_chunks, desc=f"Transcribing chunks for {date_str}")):
                    chunk_offset_sec = i * PROCESSING_CHUNK_DURATION_SEC
                    transcription_segments = transcribe_chunk_with_segments(chunk_path, whisper_model)
                    for segment in transcription_segments:
                        segment["start"] += chunk_offset_sec
                        segment["end"] += chunk_offset_sec
                        all_transcription_segments_global.append(segment)

                # --- ステップ 2.5: 最終結果のマージとエクスポート ---
                final_timeline = assign_speakers_to_transcription_segments(
                    all_transcription_segments_global, global_diarization_list
                )
                session_name = f"{date_str}_walking_log"
                merge_and_export_results(final_timeline, args.output, session_name)

            except Exception as e:
                logger.error(f"日付グループ {date_str} の処理中にエラー: {e}", exc_info=True)
            finally:
                if date_temp_dir.exists():
                    shutil.rmtree(date_temp_dir)

    finally:
        if global_temp_dir.exists():
            shutil.rmtree(global_temp_dir)

    logger.info("Walking preset processing finished.")
