#!/usr/bin/env python3
"""
文字起こしと話者分離の結果をマージし、各種形式でエクスポートする関数群
"""

import logging
from datetime import timedelta
from pathlib import Path
from typing import List, Dict, Set, Tuple
import json

from pyannote.core import Annotation, Segment

# ロガー設定
logger = logging.getLogger(__name__)


def merge_results(
    transcription_results: List[Dict],
    global_diarization: Annotation,
    output_dir: Path,
    session_name: str,
) -> Dict[str, Path]:
    """結果をマージして出力ファイル生成（改良版）"""

    merged_segments = []
    all_speakers: Set[str] = set()

    # 話者情報収集
    for turn, _, speaker_label in global_diarization.itertracks(yield_label=True):
        all_speakers.add(speaker_label)

    logger.info(f"検出された話者: {sorted(list(all_speakers))}")

    # 話者ラベルの正規化
    speaker_mapping = _create_speaker_mapping(all_speakers)
    logger.info(f"話者マッピング: {speaker_mapping}")

    # Whisperのセグメントと話者情報をマージ
    for result in transcription_results:
        start_time = result["start"]
        end_time = result["end"]
        text = result["text"].strip()

        if not text:
            continue

        # 該当時間の話者を特定
        speaker = _find_best_speaker_global(
            global_diarization, start_time, end_time, speaker_mapping
        )

        merged_segments.append(
            {"start": start_time, "end": end_time, "text": text, "speaker": speaker}
        )

    # 時間順でソート
    merged_segments.sort(key=lambda x: x["start"])

    # 同一話者の連続セグメントを統合
    merged_segments = _merge_consecutive_segments(merged_segments)

    if not merged_segments:
        logger.warning(f"セッション '{session_name}' に有効な文字起こしセグメントが見つかりませんでした。空のファイルを出力します。")


    # 出力ファイル生成
    output_files = {}

    # 必ず空でもファイルを出力
    txt_file = output_dir / f"{session_name}.txt"
    _write_txt_output(merged_segments, txt_file)
    output_files["txt"] = txt_file

    srt_file = output_dir / f"{session_name}.srt"
    _write_srt_output(merged_segments, srt_file)
    output_files["srt"] = srt_file

    vtt_file = output_dir / f"{session_name}.vtt"
    _write_vtt_output(merged_segments, vtt_file)
    output_files["vtt"] = vtt_file

    # --- ここから追加 ---
    csv_file = output_dir / f"{session_name}.csv"
    _write_csv_output(merged_segments, csv_file)
    output_files["csv"] = csv_file

    json_file = output_dir / f"{session_name}.json"
    _write_json_output(merged_segments, json_file)
    output_files["json"] = json_file
    # --- ここまで追加 ---

    return output_files


def _create_speaker_mapping(all_speakers: Set[str]) -> Dict[str, str]:
    """話者ラベルの正規化マッピング作成"""
    mapping = {}
    speaker_counter = 0

    sorted_speakers = sorted(list(all_speakers))

    for original_label in sorted_speakers:
        mapping[original_label] = f"SPEAKER_{speaker_counter}"
        speaker_counter += 1

    return mapping


def _find_best_speaker_global(
    diarization: Annotation, start_time: float, end_time: float, speaker_mapping: Dict[str, str]
) -> str:
    """グローバル話者分離結果から最適な話者を特定"""
    segment_center = start_time + (end_time - start_time) / 2
    overlapping_speakers = []

    for turn, _, speaker_label in diarization.itertracks(yield_label=True):
        overlap_start = max(turn.start, start_time)
        overlap_end = min(turn.end, end_time)
        overlap_duration = max(0, overlap_end - overlap_start)
        if overlap_duration > 0:
            overlapping_speakers.append((speaker_label, overlap_duration))

    if overlapping_speakers:
        best_speaker, _ = max(overlapping_speakers, key=lambda x: x[1])
        return speaker_mapping.get(best_speaker, f"SPEAKER_{best_speaker}")

    closest_speaker = None
    min_distance = float("inf")
    for turn, _, speaker_label in diarization.itertracks(yield_label=True):
        turn_center = turn.start + turn.duration / 2
        distance = abs(segment_center - turn_center)
        if distance < min_distance:
            min_distance = distance
            closest_speaker = speaker_label

    return speaker_mapping.get(closest_speaker, "SPEAKER_UNKNOWN") if closest_speaker else "SPEAKER_UNKNOWN"


def _merge_consecutive_segments(segments: List[Dict]) -> List[Dict]:
    """同一話者の連続セグメントを統合"""
    if not segments:
        return []
    merged = []
    current_segment = segments[0].copy()
    for next_segment in segments[1:]:
        if current_segment["speaker"] == next_segment["speaker"] and next_segment["start"] - current_segment["end"] < 1.5:
            current_segment["end"] = next_segment["end"]
            current_segment["text"] += " " + next_segment["text"]
        else:
            merged.append(current_segment)
            current_segment = next_segment.copy()
    merged.append(current_segment)
    return merged


def _format_timestamp(seconds: float) -> str:
    """タイムスタンプフォーマット (HH:MM:SS.mmm)"""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def _format_srt_timestamp(seconds: float) -> str:
    """SRT タイムスタンプフォーマット"""
    return _format_timestamp(seconds).replace(".", ",")


def _format_vtt_timestamp(seconds: float) -> str:
    """VTT タイムスタンプフォーマット"""
    return _format_timestamp(seconds)


def _write_txt_output(segments: List[Dict], output_file: Path):
    """テキスト形式で出力"""
    with open(output_file, "w", encoding="utf-8") as f:
        for segment in segments:
            start_str = _format_timestamp(segment["start"])
            end_str = _format_timestamp(segment["end"])
            f.write(f"[{start_str} - {end_str}] {segment['speaker']}: {segment['text']}\n")


def _write_srt_output(segments: List[Dict], output_file: Path):
    """SRT字幕形式で出力"""
    with open(output_file, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, 1):
            start_str = _format_srt_timestamp(segment["start"])
            end_str = _format_srt_timestamp(segment["end"])
            f.write(f"{i}\n")
            f.write(f"{start_str} --> {end_str}\n")
            f.write(f"<v {segment['speaker']}>{segment['text']}</v>\n\n")


def _write_vtt_output(segments: List[Dict], output_file: Path):
    """VTT字幕形式で出力"""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for segment in segments:
            start_str = _format_vtt_timestamp(segment["start"])
            end_str = _format_vtt_timestamp(segment["end"])
            f.write(f"{start_str} --> {end_str}\n")
            f.write(f"<v {segment['speaker']}>{segment['text']}</v>\n\n")


def _write_csv_output(segments: List[Dict], output_file: Path):
    """CSV形式で出力"""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("イベント名\n\nstart,end,speaker,text\n")
        for seg in segments:
            f.write(f"{seg['start']},{seg['end']},{seg['speaker']},{seg['text']}\n")


def _write_json_output(segments: List[Dict], output_file: Path):
    """JSON形式で出力"""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)