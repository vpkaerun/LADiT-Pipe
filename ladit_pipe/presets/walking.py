import logging
from pathlib import Path
from typing import List, Tuple
import argparse
import tempfile
import shutil

from pydub import AudioSegment
from pydub.silence import split_on_silence
import whisper
from pyannote.core import Annotation, Segment

from ladit_pipe.utils.ffmpeg_wrapper import concatenate_files, convert_to_wav
from ladit_pipe.utils.file_handler import find_audio_files
from ladit_pipe.core.transcription import transcribe_chunk
from ladit_pipe.core.export import merge_results

# ロガー設定
logger = logging.getLogger(__name__)

# 設定定数
CHUNK_LENGTH_MS = 30000  # 30秒
MAX_CHUNK_LENGTH_MS = 60000  # 60秒
MIN_SILENCE_LEN = 500  # 0.5秒
SILENCE_THRESH = -40  # dB

# ダミースピーカーラベル
SPEAKER_ME = "SPEAKER_00"


def create_intelligent_chunks(
    wav_file: Path, temp_dir: Path
) -> List[Tuple[Path, float, float]]:
    """
    音声ファイルをインテリジェントにチャンク分割します。
    無音区間と時間ベースの分割を組み合わせて、処理に適したチャンクを作成します。

    Args:
        wav_file (Path): 分割するWAVファイルのパス。
        temp_dir (Path): 中間チャンクファイルを保存する一時ディレクトリのパス。

    Returns:
        List[Tuple[Path, float, float]]: 各チャンクのファイルパス、開始時間（秒）、終了時間（秒）のタプルのリスト。
    """
    logger.info(f"チャンク分割開始: {wav_file.name}")

    # pydub で音声読み込み
    audio = AudioSegment.from_wav(wav_file)
    total_duration = len(audio)

    chunks_info = []

    if total_duration > CHUNK_LENGTH_MS:
        # 無音区間で分割
        silence_chunks = split_on_silence(
            audio,
            min_silence_len=MIN_SILENCE_LEN,
            silence_thresh=SILENCE_THRESH,
            keep_silence=200,
        )

        current_time = 0.0
        for i, chunk in enumerate(silence_chunks):
            chunk_duration = len(chunk)

            # チャンクが長すぎる場合は時間ベースで分割
            if chunk_duration > MAX_CHUNK_LENGTH_MS:
                sub_chunks = _split_by_time(chunk, CHUNK_LENGTH_MS)
                for j, sub_chunk in enumerate(sub_chunks):
                    sub_chunk_file = temp_dir / f"{wav_file.stem}_chunk_{i}_{j}.wav"
                    sub_chunk.export(sub_chunk_file, format="wav")

                    start_time = current_time
                    end_time = current_time + len(sub_chunk) / 1000.0
                    chunks_info.append((sub_chunk_file, start_time, end_time))
                    current_time = end_time
            else:
                chunk_file = temp_dir / f"{wav_file.stem}_chunk_{i}.wav"
                chunk.export(chunk_file, format="wav")

                start_time = current_time
                end_time = current_time + chunk_duration / 1000.0
                chunks_info.append((chunk_file, start_time, end_time))
                current_time = end_time
    else:
        # 短いファイルはそのまま使用
        chunks_info.append((wav_file, 0.0, total_duration / 1000.0))

    logger.info(f"チャンク分割完了: {len(chunks_info)}個のチャンク")
    return chunks_info


def _split_by_time(audio: AudioSegment, chunk_length_ms: int) -> List[AudioSegment]:
    """
    AudioSegmentオブジェクトを、指定されたミリ秒単位のチャンク長で分割します。

    Args:
        audio (AudioSegment): 分割するAudioSegmentオブジェクト。
        chunk_length_ms (int): 各チャンクの長さ（ミリ秒）。

    Returns:
        List[AudioSegment]: 分割されたAudioSegmentオブジェクトのリスト。
    """
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i : i + chunk_length_ms]
        chunks.append(chunk)
    return chunks


def run(args: argparse.Namespace):
    """
    ウォーキングプリセットの処理を実行します。
    複数の入力ファイルを連結し、話者分離をスキップして高速に文字起こしを行います。

    Args:
        args (argparse.Namespace): コマンドラインからパースされた引数オブジェクト。
    """
    logger.info("Walking preset processing started.")

    temp_dir = Path(tempfile.mkdtemp())
    logger.info(f"一時ディレクトリを作成しました: {temp_dir}")

    try:
        # 1. 入力ファイルの検索とWAV変換
        input_audio_files = find_audio_files(args.input)
        if not input_audio_files:
            logger.error(
                f"入力ディレクトリ {args.input} に処理可能なオーディオファイルが見つかりませんでした。"
            )
            return

        converted_files = []
        for i, file_path in enumerate(input_audio_files):
            wav_output_path = temp_dir / f"converted_{file_path.stem}.wav"
            converted_files.append(convert_to_wav(file_path, wav_output_path))

        # 2. 複数ファイルを連結
        combined_wav_path = temp_dir / "combined.wav"
        concatenate_files(converted_files, combined_wav_path)
        logger.info(f"すべての入力ファイルを {combined_wav_path.name} に連結しました。")

        # 3. Whisperモデルのロード
        logger.info(
            f"Whisperモデル {args.whisper_model} をロード中... (デバイス: {args.device})"
        )
        whisper_model = whisper.load_model(args.whisper_model, device=args.device)
        logger.info("Whisperモデルのロードが完了しました。")

        # 4. チャンク分割
        chunks_info = create_intelligent_chunks(combined_wav_path, temp_dir)
        logger.info(f"合計 {len(chunks_info)} 個のチャンクに分割しました。")

        transcription_results = []
        for i, (chunk_file, start_time, end_time) in enumerate(chunks_info):
            logger.info(
                f"チャンク {i+1}/{len(chunks_info)} を文字起こし中 ({start_time:.2f}s - {end_time:.2f}s)..."
            )
            # 話者分離をスキップするため、ダミーのAnnotationオブジェクトを生成
            dummy_diarization = Annotation(uri="combined_audio")
            dummy_diarization[Segment(start_time, end_time)] = SPEAKER_ME

            chunk_transcription = transcribe_chunk(
                audio_file=chunk_file,
                diarization=dummy_diarization,
                whisper_model=whisper_model,
                min_speakers=args.min_speakers,  # walking presetでは使われないが引数として渡す
                max_speakers=args.max_speakers,  # walking presetでは使われないが引数として渡す
            )
            transcription_results.extend(chunk_transcription)
        logger.info("すべてのチャンクの文字起こしが完了しました。")

        # 5. 結果のマージとエクスポート
        output_base_name = (
            args.input.stem if args.input.is_file() else "combined_output"
        )
        merge_results(transcription_results, args.output, output_base_name)
        logger.info(f"結果を {args.output} にエクスポートしました。")

    finally:
        # 一時ディレクトリのクリーンアップ
        shutil.rmtree(temp_dir)
        logger.info(f"一時ディレクトリ {temp_dir} を削除しました。")

    logger.info("Walking preset processing finished.")
