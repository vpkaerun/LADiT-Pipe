# ladit_pipe/presets/meeting.py (Load-Balanced Phoenix Architecture - FINAL)

import logging
import argparse
import tempfile
import shutil
import gc
from pathlib import Path
from ladit_pipe.utils.file_handler import find_audio_files

import torch
import whisper
from tqdm import tqdm

# LADiT-Pipeの内部モジュールをインポート
from ladit_pipe.utils.ffmpeg_wrapper import (
    convert_to_wav_with_progress,
    split_into_chunks_with_progress,
    concatenate_files,
)
from ladit_pipe.core.diarization import (
    perform_chunk_diarization,
)  # 新しい関数名
from ladit_pipe.core.transcription import (
    transcribe_chunk_with_segments,
)  # 新しい関数名
from ladit_pipe.core.export import (
    merge_and_export_results,
    assign_speakers_to_transcription_segments,
)  # 新しい関数名

# ロガー設定
logger = logging.getLogger(__name__)

# 処理チャンクの長さ（秒）
PROCESSING_CHUNK_DURATION_SEC = 600  # 10分



from ladit_pipe.utils.file_handler import find_audio_files


def run(args: argparse.Namespace):
    """
    ミーティングプリセット（最終負荷分散版）を実行します。
    """
    logger.info(
        "Meeting preset [Load-Balanced Phoenix Architecture] "
        "processing started."
    )
    temp_dir = Path(tempfile.mkdtemp(prefix="ladit_pipe_final_"))
    logger.debug(f"一時ディレクトリ: {temp_dir}")

    try:
        input_path = args.input

        if input_path.is_dir():
            logger.info(f"入力パス {input_path.name} はディレクトリです。")
            all_audio_files = find_audio_files(input_path)
            if not all_audio_files:
                logger.warning(
                    f"指定されたパス {input_path} に"
                    "音声ファイルが見つかりませんでした。"
                )
                return

            converted_files_for_concat = []
            for original_file in all_audio_files:
                converted_file = convert_to_wav_with_progress(
                    original_file, temp_dir
                )
                if converted_file:
                    converted_files_for_concat.append(converted_file)
                else:
                    logger.error(
                        f"ファイル {original_file.name} のWAV変換に失敗しました。"
                        "このファイルをスキップします。"
                    )

            if not converted_files_for_concat:
                logger.error(
                    f"入力ディレクトリ {input_path} に連結可能なファイルが見つかりませんでした。"
                )
                return

            concatenated_output_path = temp_dir / "concatenated_session.wav"
            wav_file = concatenate_files(
                converted_files_for_concat, concatenated_output_path
            )
            logger.info(
                f"連結されたファイルを {concatenated_output_path.name} に保存しました。"
            )

        else:
            logger.info(f"入力ファイル: {input_path.name}")
            wav_file = convert_to_wav_with_progress(input_path, temp_dir)
            if not wav_file:
                return

        # --- ステップ 2: 巨大WAVを処理可能な中間チャンクに分割 ---
        processing_chunks = split_into_chunks_with_progress(
            wav_file, temp_dir, PROCESSING_CHUNK_DURATION_SEC
        )

        # --- ステップ 3: モデルの事前ロード ---
        whisper_model = whisper.load_model(
            args.whisper_model, device=args.device
        )
        # Diarizationモデルは、diarizationモジュール内でロードされる

        # --- ステップ 4: チャンクごとの、分離→文字起こしパイプライン ---
        all_transcription_segments_global = []
        all_diarization_segments_global = []

        for i, chunk_path in enumerate(
            tqdm(processing_chunks, desc="Processing Chunks")
        ):
            chunk_offset_sec = i * PROCESSING_CHUNK_DURATION_SEC
            logger.info(f"Processing chunk {i+1}/{len(processing_chunks)}...")

            # 4a: この10分チャンクの話者分離
            chunk_diarization_annotation = perform_chunk_diarization(
                chunk_path,
                args.hf_token,
                args.device,
                args.min_speakers,
                args.max_speakers,
            )

            # 【！】ここが、最後の、そして最も重要な、翻訳処理
            # Annotationオブジェクトを、辞書のリストに変換する
            chunk_diarization_list = []
            for segment, _, speaker in chunk_diarization_annotation.itertracks(yield_label=True):
                chunk_diarization_list.append({
                    "start": segment.start,
                    "end": segment.end,
                    "speaker": speaker,
                })

            # 変換後のリストを、グローバルなタイムラインに追加
            for entry in chunk_diarization_list:
                entry["start"] += chunk_offset_sec
                entry["end"] += chunk_offset_sec
                all_diarization_segments_global.append(entry)

            # 4b: この10分チャンクの文字起こし (セグメント付き)
            transcription_segments = transcribe_chunk_with_segments(
                chunk_path, whisper_model
            )
            for segment in transcription_segments:
                segment["start"] += chunk_offset_sec
                segment["end"] += chunk_offset_sec
                all_transcription_segments_global.append(segment)

            # メモリ解放
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # --- ステップ 5: 全体の結果をマージし、話者を割り当てる ---
        logger.info("全体の文字起こしと話者分離結果をマージします...")
        final_timeline = assign_speakers_to_transcription_segments(
            all_transcription_segments_global, all_diarization_segments_global
        )

        # --- ステップ 5: 最終エクスポート ---
        logger.info("最終結果をファイルに出力します...")

        # 【！】ここが、最後の、そして最も重要な、仕上げ
        # セッション名を、入力パスがディレクトリかファイルかで、賢く決定する
        if input_path.is_dir():
            # ディレクトリの場合、その中にある「最初のファイル名」をベースにする
            # find_audio_filesはソート済みのリストを返すので、これでOK
            first_file = find_audio_files(input_path)[0]
            session_name = f"{first_file.stem}_session"
        else:
            # ファイルの場合、そのファイル名をベースにする
            session_name = input_path.stem

        logger.info(f"セッション名: {session_name}")

        merge_and_export_results(
            final_timeline, args.output, session_name
        )

    except Exception as e:
        logger.error(f"処理中に致命的なエラー: {e}", exc_info=True)

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logger.debug(f"一時ディレクトリを削除しました。")

    logger.info("Load-Balanced Phoenix Architecture processing finished.")
