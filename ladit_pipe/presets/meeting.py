import logging
from tqdm import tqdm
from pathlib import Path
import argparse
import tempfile
import shutil
from typing import List, Dict, Any
import gc
from datetime import timedelta

import numpy as np
from scipy.spatial.distance import cosine
import whisper
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
import torch  # Ensure torch is imported
from pyannote.audio import Model
from pyannote.audio.core.io import Audio  # Audioローダーを直接インポート
from pydub import AudioSegment

from ladit_pipe.utils.ffmpeg_wrapper import (
    get_duration_sec,
    split_into_chunks,
    convert_to_wav,
    extract_clip,
)
from ladit_pipe.utils.file_handler import find_audio_files
from ladit_pipe.core.diarization import diarize_chunk
from ladit_pipe.core.transcription import transcribe_chunk
from ladit_pipe.core.export import export_from_timeline

# ロガー設定
logger = logging.getLogger(__name__)

# 定数
LONG_CHUNK_DURATION_SEC = 30  # 30秒
COSINE_THRESHOLD = 0.8  # コサイン類似度の閾値


def run(args: argparse.Namespace):
    """
    ミーティングプリセットの処理を実行します。
    Diarizationパスのロジックを実装し、複数の長時間ファイルをVRAMの制約を受けずに処理するため、
    ディスクベースでチャンク分割を行い、ファイル間で一貫した話者IDを割り当てる基盤を構築します。

    Args:
        args (argparse.Namespace): コマンドラインからパースされた引数オブジェクト。
    """
    logger.info("Meeting preset processing started.")

    temp_dir = Path(tempfile.mkdtemp())
    logger.info(f"一時ディレクトリを作成しました: {temp_dir}")

    try:
        # 1. モデルの初期化 (フェーズ1: Diarization & Embedding)
        logger.info("Pyannote Speaker Diarization パイプラインをロード中...")
        # Hugging Faceトークンが必要な場合があるため、argsから取得
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization@2.1", use_auth_token=args.hf_token
        )
        diarization_pipeline.to(
            torch.device(args.device)
        )  # デバイスにパイプラインを移動
        logger.info("Pyannote Speaker Diarization パイプラインのロードが完了しました。")

        logger.info("Pyannote Embedding モデルをロード中...")
        embedding_model = Model.from_pretrained(
            "pyannote/embedding", use_auth_token=args.hf_token
        )
        if embedding_model is None:
            logger.error(
                "Pyannote Embedding モデルのロードに失敗しました。"
                "Hugging Faceトークンが正しいか、モデルの利用規約に同意しているか確認してください。"
            )
            return
        embedding_model.to(torch.device(args.device))
        logger.info("Pyannote Embedding モデルのロードが完了しました。")

        # 2. 入力ファイルの検索とWAV変換
        input_audio_files = find_audio_files(args.input)
        if not input_audio_files:
            logger.error(
                f"入力ディレクトリ {args.input} に処理可能な"
                "オーディオファイルが見つかりませんでした。"
            )
            return

        converted_files_info: List[Dict[str, Any]] = []
        for i, file_path in enumerate(input_audio_files):
            wav_output_path = temp_dir / f"converted_{file_path.stem}.wav"
            converted_wav_path = convert_to_wav(file_path, wav_output_path)
            converted_files_info.append(
                {"original_path": file_path, "wav_path": converted_wav_path}
            )
        logger.info(f"全入力ファイルをWAV形式に変換しました。")

        # 3. グローバルタイムラインの構築
        file_offsets: List[Dict[str, Any]] = []
        current_global_offset = 0.0
        for file_info in converted_files_info:
            duration = get_duration_sec(file_info["wav_path"])
            file_offsets.append(
                {
                    "original_path": file_info["original_path"],
                    "wav_path": file_info["wav_path"],
                    "global_start_sec": current_global_offset,
                    "global_end_sec": current_global_offset + duration,
                    "duration_sec": duration,
                }
            )
            current_global_offset += duration
        logger.info("グローバルタイムラインを構築しました。")
        for offset_info in file_offsets:
            logger.info(
                f"  ファイル: {offset_info['original_path'].name}, "
                f"グローバル開始: {offset_info['global_start_sec']:.2f}s, "
                f"グローバル終了: {offset_info['global_end_sec']:.2f}s"
            )

        # 4. チャンク分割 (15分単位のlong_chunk)
        all_long_chunks_info: List[Dict[str, Any]] = []
        for file_info in file_offsets:
            chunks_for_file = split_into_chunks(
                file_info["wav_path"], temp_dir, LONG_CHUNK_DURATION_SEC
            )

            # 各チャンクの元のファイル内での開始時間を計算し、情報を保持
            for i, chunk_file in enumerate(chunks_for_file):
                chunk_start_in_original_sec = i * LONG_CHUNK_DURATION_SEC
                all_long_chunks_info.append(
                    {
                        "chunk_path": chunk_file,
                        "original_file_wav_path": file_info["wav_path"],
                        "global_offset_of_original_file": file_info["global_start_sec"],
                        "chunk_start_in_original_sec": chunk_start_in_original_sec,
                    }
                )
        logger.info(
            f"全入力ファイルを {LONG_CHUNK_DURATION_SEC}秒単位のチャンクに分割しました。"
            f"合計 {len(all_long_chunks_info)} 個のチャンク。"
        )

        # 5. 話者分離の実行 (チャンクごと)
        diarization_results_per_chunk: List[Any] = (
            []
        )  # pyannote.core.Annotation オブジェクトのリストを想定
        for i, chunk_info in enumerate(all_long_chunks_info):
            chunk_file = chunk_info["chunk_path"]
            logger.info(
                f"チャンク {i+1}/{len(all_long_chunks_info)} の話者分離を実行中: "
                f"{chunk_file.name}"
            )

            diarization_output = diarize_chunk(
                chunk_file=chunk_file,
                diarization_pipeline=diarization_pipeline,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
            )
            diarization_results_per_chunk.append(diarization_output)
            del diarization_output
            torch.cuda.empty_cache()
            gc.collect()
        logger.info("チャンクごとの話者分離が完了しました。")

        # === 新しいアーキテクチャ START ===

        # 6. 話者ごとの音声セグメントを収集
        logger.info("話者ごとの音声セグメントを収集します。")
        speaker_audio_segments: Dict[str, AudioSegment] = {}

        for i, diarization_output in enumerate(tqdm(diarization_results_per_chunk, desc="Collecting speaker segments")):
            chunk_info = all_long_chunks_info[i]
            chunk_audio = AudioSegment.from_wav(chunk_info["chunk_path"])
            
            for segment, _, speaker in diarization_output.itertracks(yield_label=True):
                start_ms = int(segment.start * 1000)
                end_ms = int(segment.end * 1000)
                turn_audio = chunk_audio[start_ms:end_ms]
                
                if speaker not in speaker_audio_segments:
                    speaker_audio_segments[speaker] = turn_audio
                else:
                    speaker_audio_segments[speaker] += turn_audio

        # 7. 話者ごとに分離した音声ファイルを出力し、文字起こし
        logger.info("話者ごとに分離した音声ファイルを文字起こしします。")
        global_timeline = []
        whisper_model = whisper.load_model(args.whisper_model, device=args.device)

        for speaker, full_audio in tqdm(speaker_audio_segments.items(), desc="Transcribing speakers"):
            speaker_wav_path = temp_dir / f"speaker_{speaker}.wav"
            full_audio.export(speaker_wav_path, format="wav")
            
            logger.info(f"話者 {speaker} の音声を文字起こし中...")
            transcription_result = whisper_model.transcribe(str(speaker_wav_path), language="ja")
            
            # Whisperが生成したタイムスタンプ付きセグメントをグローバルタイムラインに追加
            for seg in transcription_result["segments"]:
                # 注意：ここでのタイムスタンプは、連結された音声内でのローカルなもの。
                # 真のグローバルタイムスタンプへの再マッピングは、より高度な実装が必要。
                # 今回は、まずこの方法で機能するかを確認する。
                global_timeline.append({
                    "start": seg["start"], # 仮のタイムスタンプ
                    "end": seg["end"],     # 仮のタイムスタンプ
                    "speaker": speaker,
                    "text": seg["text"]
                })

        # タイムラインをソート
        global_timeline.sort(key=lambda x: x["start"])

        # === 新しいアーキテクチャ END ===

        logger.info("文字起こしが完了しました。")

        logger.info("Whisper modelをメモリから解放します...")
        del whisper_model  # whisper_modelをロードした変数名に合わせてください
        torch.cuda.empty_cache()
        gc.collect()

        # 8. 結果のエクスポート (Export Pass)
        logger.info("結果のエクスポートを開始します。")
        session_name = args.input.stem if args.input.is_file() else "session_output"
        export_from_timeline(
            timeline=global_timeline,
            output_dir=args.output,
            session_name=session_name,
            source_files_info=file_offsets,
        )
        logger.info("結果のエクスポートが完了しました。")

    finally:
        # 一時ディレクトリのクリーンアップ
        shutil.rmtree(temp_dir)
        logger.debug(f"一時ディレクトリ {temp_dir} を削除しました。")

    logger.info("Meeting preset processing finished.")
