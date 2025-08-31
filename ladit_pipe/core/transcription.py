#!/usr/bin/env python3
"""
文字起こしに関連する関数群
"""

import logging
import os
import re
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import whisper
from pydub import AudioSegment
from pydub.silence import split_on_silence

# ロガー設定
logger = logging.getLogger(__name__)

# 設定定数
CHUNK_LENGTH_MS = 30000  # 30秒（文字起こし用）
MIN_SILENCE_LEN = 500
SILENCE_THRESH = -40


def create_transcription_chunks(wav_file: Path, temp_dir: Path) -> List[Tuple[Path, float, float]]:
    """文字起こし用の小さなチャンク作成"""
    logger.info(f"文字起こし用チャンク作成: {wav_file.name}")

    audio = AudioSegment.from_wav(wav_file)
    total_duration = len(audio)
    chunks_info = []

    if total_duration > CHUNK_LENGTH_MS:
        # 無音区間で分割
        silence_chunks = split_on_silence(
            audio,
            min_silence_len=MIN_SILENCE_LEN,
            silence_thresh=SILENCE_THRESH,
            keep_silence=200
        )

        current_time = 0.0
        for i, chunk in enumerate(silence_chunks):
            chunk_duration = len(chunk)

            # チャンクが長すぎる場合は時間ベースで分割
            if chunk_duration > CHUNK_LENGTH_MS * 2:
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
        chunks_info.append((wav_file, 0.0, total_duration / 1000.0))

    logger.info(f"文字起こしチャンク作成完了: {len(chunks_info)}個")
    return chunks_info


def _split_by_time(audio: AudioSegment, chunk_length_ms: int) -> List[AudioSegment]:
    """時間ベースでオーディオを分割"""
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunks.append(chunk)
    return chunks


def transcribe_chunk(chunk_file: Path, whisper_model: whisper.Whisper) -> dict:
    """チャンクの文字起こし"""
    # テスト環境変数 'LADIT_PIPE_TESTING' が設定されている場合、モックデータを返す
    if os.environ.get("LADIT_PIPE_TESTING"):
        logger.info(f"テストモード: {chunk_file.name} のダミー文字起こし結果を返します。")
        return {
            "text": "これはテスト用の文字起こしです。",
            "segments": [
                {"start": 0.5, "end": 4.5, "text": "これはテスト用の"},
                {"start": 5.0, "end": 9.5, "text": "文字起こしです。"},
            ],
        }

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        options = {
            "language": "ja",
            "task": "transcribe",
            "fp16": True,
            "temperature": 0.0,
            "best_of": 1,
            "beam_size": 1,
            "patience": 1.0,
            "suppress_tokens": [-1],
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.0,
            "condition_on_previous_text": False,
        }

        result = whisper_model.transcribe(str(chunk_file), **options)
        result = _post_process_transcription(result)
        return result
    except Exception as e:
        logger.error(f"文字起こしエラー {chunk_file}: {e}")
        return {}


def _post_process_transcription(result: Dict) -> Dict:
    """転写結果の後処理"""
    if "segments" not in result:
        return result

    processed_segments = []

    for segment in result["segments"]:
        text = segment.get("text", "").strip()
        cleaned_text = _remove_repetitions(text)

        if len(cleaned_text) > 0 and not _is_meaningless_segment(cleaned_text):
            segment["text"] = cleaned_text
            processed_segments.append(segment)

    result["segments"] = processed_segments
    result["text"] = " ".join([seg["text"] for seg in processed_segments])

    return result


def _remove_repetitions(text: str) -> str:
    """繰り返しパターンを除去"""
    # 同一単語の連続繰り返し除去
    text = re.sub(r'(\S+?)(\1){2,}', r'\1', text)

    words = text.split()
    if len(words) > 6:
        i = 0
        cleaned_words = []
        while i < len(words):
            # 2語フレーズの繰り返しチェック
            if i + 3 < len(words) and words[i:i+2] == words[i+2:i+4]:
                cleaned_words.extend(words[i:i+2])
                j = i + 4
                while j + 1 < len(words) and words[i:i+2] == words[j:j+2]:
                    j += 2
                i = j
            elif i + 5 < len(words) and words[i:i+3] == words[i+3:i+6]:
                cleaned_words.extend(words[i:i+3])
                j = i + 6
                while j + 2 < len(words) and words[i:i+3] == words[j:j+3]:
                    j += 3
                i = j
            else:
                cleaned_words.append(words[i])
                i += 1

        text = " ".join(cleaned_words)

    text = re.sub(r'(\S)\1{4,}', r'\1', text)
    return text.strip()


def _is_meaningless_segment(text: str) -> bool:
    """意味のないセグメントかどうか判定"""
    if len(text.strip()) < 2:
        return True
    if re.match(r'^(\S)\1*$', text.strip()):
        return True
    if re.match(r'^[^\w\s]*$', text.strip()):
        return True
    return False


def _post_process_transcription(result: Dict) -> Dict:
    """転写結果の後処理"""
    if "segments" not in result:
        return result

    processed_segments = []

    for segment in result["segments"]:
        text = segment.get("text", "").strip()
        cleaned_text = _remove_repetitions(text)

        if len(cleaned_text) > 0 and not _is_meaningless_segment(cleaned_text):
            segment["text"] = cleaned_text
            processed_segments.append(segment)

    result["segments"] = processed_segments
    result["text"] = " ".join([seg["text"] for seg in processed_segments])

    return result


def _remove_repetitions(text: str) -> str:
    """繰り返しパターンを除去"""
    # 同一単語の連続繰り返し除去
    text = re.sub(r'(\S+?)(\1){2,}', r'\1', text)

    words = text.split()
    if len(words) > 6:
        i = 0
        cleaned_words = []
        while i < len(words):
            # 2語フレーズの繰り返しチェック
            if i + 3 < len(words) and words[i:i+2] == words[i+2:i+4]:
                cleaned_words.extend(words[i:i+2])
                j = i + 4
                while j + 1 < len(words) and words[i:i+2] == words[j:j+2]:
                    j += 2
                i = j
            elif i + 5 < len(words) and words[i:i+3] == words[i+3:i+6]:
                cleaned_words.extend(words[i:i+3])
                j = i + 6
                while j + 2 < len(words) and words[i:i+3] == words[j:j+3]:
                    j += 3
                i = j
            else:
                cleaned_words.append(words[i])
                i += 1

        text = " ".join(cleaned_words)

    text = re.sub(r'(\S)\1{4,}', r'\1', text)
    return text.strip()


def _is_meaningless_segment(text: str) -> bool:
    """意味のないセグメントかどうか判定"""
    if len(text.strip()) < 2:
        return True
    if re.match(r'^(\S)\1*$', text.strip()):
        return True
    if re.match(r'^[^\w\s]*$', text.strip()):
        return True
    return False