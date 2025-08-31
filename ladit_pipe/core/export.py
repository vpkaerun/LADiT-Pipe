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

def merge_and_export_results(
    final_timeline: List[Dict],
    output_dir: Path,
    session_name: str,
) -> Dict[str, Path]:
    """最終結果をマージして出力ファイル生成"""

    merged_segments = []
    all_speakers: Set[str] = set()

    

    # 同一話者の連続セグメントを統合
    # 出力ファイル生成
    output_files = {}

    # 必ず空でもファイルを出力
    txt_file = output_dir / f"{session_name}.txt"
    _write_txt_output(final_timeline, txt_file)
    output_files["txt"] = txt_file

    srt_file = output_dir / f"{session_name}.srt"
    _write_srt_output(final_timeline, srt_file)
    output_files["srt"] = srt_file

    vtt_file = output_dir / f"{session_name}.vtt"
    _write_vtt_output(final_timeline, vtt_file)
    output_files["vtt"] = vtt_file

    # --- ここから追加 ---
    csv_file = output_dir / f"{session_name}.csv"
    _write_csv_output(final_timeline, csv_file)
    output_files["csv"] = csv_file

    json_file = output_dir / f"{session_name}.json"
    _write_json_output(final_timeline, json_file)
    output_files["json"] = json_file
    # --- ここまで追加 ---

    return output_files


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