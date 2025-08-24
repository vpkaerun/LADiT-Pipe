import logging
from pathlib import Path
from typing import List, Dict, Tuple
from pyannote.core import Annotation
from datetime import timedelta

# ロガー設定
logger = logging.getLogger(__name__)


def _create_speaker_mapping(all_speakers: set) -> Dict[str, str]:
    """
    話者ラベルの正規化マッピングを作成します。
    元の話者ラベル（例: 'SPEAKER_00'）を 'SPEAKER_0', 'SPEAKER_1'のように連番にマッピングします。

    Args:
        all_speakers (set): 検出された全ての話者ラベルのセット。

    Returns:
        Dict[str, str]: 元ラベルから正規化ラベルへのマッピング辞書。
    """
    mapping = {}
    speaker_counter = 0

    # 数値でソートして一貫性を保つ
    sorted_speakers = sorted(all_speakers)

    for original_label in sorted_speakers:
        mapping[original_label] = f"SPEAKER_{speaker_counter}"
        speaker_counter += 1

    return mapping


def _find_best_speaker(
    diarization: Annotation,
    start_time: float,
    end_time: float,
    chunk_start: float,
    speaker_mapping: Dict[str, str],
) -> str:
    """
    指定された時間範囲に最も適した話者を話者分離結果から特定します。
    重複する話者がいる場合は、最も重複時間の長い話者を選択します。
    重複がない場合は、最も近い話者を選択します。

    Args:
        diarization (Annotation): チャンクの話者分離結果、Annotationオブジェクト。
        start_time (float): 検索する時間範囲の開始時間（秒）。
        end_time (float): 検索する時間範囲（秒）。
        chunk_start (float): 元の音声ファイルにおけるチャンクの開始時間（秒）。
        speaker_mapping (Dict[str, str]): 話者ラベルの正規化マッピング辞書。

    Returns:
        str: 特定された話者の正規化されたラベル。見つからない場合は "SPEAKER_UNKNOWN"。
    """
    # チャンク内の相対時間に変換
    relative_start = start_time - chunk_start
    relative_end = end_time - chunk_start
    segment_center = relative_start + (relative_end - relative_start) / 2

    # 重複の多い話者を特定
    overlapping_speakers = []

    for turn, _, speaker_label in diarization.itertracks(yield_label=True):
        # 重複時間を計算
        overlap_start = max(turn.start, relative_start)
        overlap_end = min(turn.end, relative_end)
        overlap_duration = max(0, overlap_end - overlap_start)

        if overlap_duration > 0:
            overlapping_speakers.append((speaker_label, overlap_duration, turn))

    if overlapping_speakers:
        # 最大重複時間の話者を選択
        best_speaker, max_overlap, _ = max(overlapping_speakers, key=lambda x: x[1])
        return speaker_mapping.get(best_speaker, f"SPEAKER_{best_speaker}")

    # 重複がない場合、最も近い話者を選択
    closest_speaker = None
    min_distance = float("inf")

    for turn, _, speaker_label in diarization.itertracks(yield_label=True):
        # セグメント中心からの距離を計算
        turn_center = turn.start + turn.duration / 2
        distance = abs(segment_center - turn_center)

        if distance < min_distance:
            min_distance = distance
            closest_speaker = speaker_label

    if closest_speaker:
        return speaker_mapping.get(closest_speaker, f"SPEAKER_{closest_speaker}")

    return "SPEAKER_UNKNOWN"


def _merge_consecutive_segments(segments: List[Dict]) -> List[Dict]:
    """
    同一話者の連続するセグメントを統合します。
    時間的に近接している（1.5秒以内）同一話者のセグメントを結合し、テキストも連結します。

    Args:
        segments (List[Dict]): 処理対象のセグメントのリスト。
                            各セグメントは'start', 'end', 'text', 'speaker'を含む辞書。

    Returns:
        List[Dict]: 統合されたセグメントのリスト。
    """
    if not segments:
        return segments

    merged = []
    current_segment = segments[0].copy()

    for next_segment in segments[1:]:
        # 同一話者で時間が近接している場合は統合
        time_gap = next_segment["start"] - current_segment["end"]
        same_speaker = current_segment["speaker"] == next_segment["speaker"]
        # より短い間隔で統合（話者変化を保持するため）
        close_in_time = time_gap < 1.5  # 1.5秒以内の間隔（短縮）

        if same_speaker and close_in_time:
            # セグメントを統合
            current_segment["end"] = next_segment["end"]
            # テキスト統合時に重複を避ける
            if current_segment["text"].strip() and next_segment["text"].strip():
                current_segment["text"] += " " + next_segment["text"]
            elif next_segment["text"].strip():
                current_segment["text"] = next_segment["text"]
        else:
            # 現在のセグメントを完了し、次のセグメントを開始
            merged.append(current_segment)
            current_segment = next_segment.copy()

    # 最後のセグメントを追加
    merged.append(current_segment)

    return merged

# ロガー設定
logger = logging.getLogger(__name__)


def _format_timestamp(seconds: float) -> str:
    """
    秒数を 'HH:MM:SS.mmm' 形式のタイムスタンプ文字列にフォーマットします。

    Args:
        seconds (float): フォーマットする秒数。

    Returns:
        str: フォーマットされたタイムスタンプ文字列。
    """
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def _format_srt_timestamp(seconds: float) -> str:
    """
    秒数をSRT形式のタイムスタンプ文字列にフォーマットします（カンマ区切り）。

    Args:
        seconds (float): フォーマットする秒数。

    Returns:
        str: SRT形式でフォーマットされたタイムスタンプ文字列。
    """
    return _format_timestamp(seconds).replace(".", ",")


def _format_vtt_timestamp(seconds: float) -> str:
    """
    秒数をVTT形式のタイムスタンプ文字列にフォーマットします（ピリオド区切り）。

    Args:
        seconds (float): フォーマットする秒数。

    Returns:
        str: VTT形式でフォーマットされたタイムスタンプ文字列。
    """
    return _format_timestamp(seconds)


def _write_txt_output(segments: List[Dict], output_file: Path):
    """
    セグメントデータをプレーンテキスト形式でファイルに書き込みます。
    各行は '[開始時間 - 終了時間] 話者: テキスト' の形式です。

    Args:
        segments (List[Dict]): 書き込むセグメントのリスト。
        output_file (Path): 出力ファイルのパス。
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for segment in segments:
            start_str = _format_timestamp(segment["start"])
            end_str = _format_timestamp(segment["end"])
            f.write(
                f"[{start_str} - {end_str}] "
                f"{segment['speaker']}: {segment['text']}\n")
    logger.info(f"テキストファイルを出力しました: {output_file.name}")


def _write_srt_output(segments: List[Dict], output_file: Path):
    """
    セグメントデータをSRT字幕形式でファイルに書き込みます。

    Args:
        segments (List[Dict]): 書き込むセグメントのリスト。
        output_file (Path): 出力ファイルのパス。
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, 1):
            start_str = _format_srt_timestamp(segment["start"])
            end_str = _format_srt_timestamp(segment["end"])
            f.write(f"{i}\n")
            f.write(f"{start_str} --> {end_str}\n")
            f.write(
                f"<v {segment['speaker']}>\n"
                f"{segment['text']}\n\n")
    logger.info(f"SRTファイルを出力: {output_file.name}")


def _write_vtt_output(segments: List[Dict], output_file: Path):
    """
    セグメントデータをVTT字幕形式でファイルに書き込みます。

    Args:
        segments (List[Dict]): 書き込むセグメントのリスト。
        output_file (Path): 出力ファイルのパス。
    """
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for segment in segments:
            start_str = _format_vtt_timestamp(segment["start"])
            end_str = _format_vtt_timestamp(segment["end"])
            f.write(f"{start_str} --> {end_str}\n")
            f.write(
                f"<v {segment['speaker']}> {segment['text']}\n\n")
    logger.info(f"VTTファイルを出力しました: {output_file.name}")


def merge_results(
    chunk_results: List[Tuple[Dict, Annotation, float, float]],
    output_base: Path,
) -> Dict[str, Path]:
    """
    話者分離と文字起こしのチャンク結果をマージし、最終的な出力ファイルを生成します。
    テキスト、SRT、VTT形式で出力します。

    Args:
        chunk_results: List[Tuple[Dict, Annotation, float, float]],
        # 各チャンクの文字起こし結果、話者分離結果、開始時間、終了時間のタプルのリスト。
        output_base (Path): 出力ファイルのベースパス（ファイル名とディレクトリを含む）。

    Returns:
        Dict[str, Path]: 生成された出力ファイルの形式とパスのマッピング辞書。
    """
    merged_segments = []
    all_speakers = set()

    # まず全チャンクから話者情報を収集
    for transcription, diarization, chunk_start, chunk_end in chunk_results:
        for turn, _, speaker_label in diarization.itertracks(yield_label=True):
            all_speakers.add(speaker_label)

    logger.info(f"検出された話者: {sorted(all_speakers)}")

    # 話者ラベルの正規化マッピング作成
    speaker_mapping = _create_speaker_mapping(all_speakers)
    logger.info(f"話者マッピング: {speaker_mapping}")

    # Whisperのセグメントと話者情報をマージ
    for transcription, diarization, chunk_start, chunk_end in chunk_results:
        for segment in transcription.get("segments", []):
            start_time = chunk_start + segment["start"]
            end_time = chunk_start + segment["end"]
            text = segment["text"].strip()

            if not text:
                continue

            # 該当時間の話者を特定
            speaker = _find_best_speaker(
                diarization, start_time, end_time, chunk_start, speaker_mapping)

            merged_segments.append(
                {"start": start_time, "end": end_time, "text": text, "speaker": speaker}
            )

    # 時間順でソート
    merged_segments.sort(key=lambda x: x["start"])

    # 同一話者の連続セグメントを統合
    merged_segments = _merge_consecutive_segments(merged_segments)

    # 出力ファイル生成
    output_files = {}

    # 安全なファイル名生成（時刻を含むファイル名に対応）
    base_name = output_base.stem
    output_dir = output_base.parent

    # テキスト形式
    txt_file = output_dir / f"{base_name}_diarized.txt"
    _write_txt_output(merged_segments, txt_file)
    output_files["txt"] = txt_file

    # SRT 形式
    srt_file = output_dir / f"{base_name}.srt"
    _write_srt_output(merged_segments, srt_file)
    output_files["srt"] = srt_file

    # VTT 形式
    vtt_file = output_dir / f"{base_name}.vtt"
    _write_vtt_output(merged_segments, output_file)
    output_files["vtt"] = {}

    return output_files


def export_from_timeline(
    timeline: List[Dict],
    output_dir: Path,
    session_name: str,
    source_files_info: List[Dict],
):
    """
    統一された話者IDと文字起こし結果を含むグローバルタイムラインを元に、
    セッション全体のテキストファイルと、各入力ファイルごとのSRT字幕ファイルを生成します。

    Args:
        timeline (List[Dict]): グローバルな話者IDと文字起こし結果を含むセグメントのリスト。
        output_dir (Path): 出力ファイルを保存するディレクトリのパス。
        session_name (str): セッション全体の出力ファイルに使用するベース名。
        source_files_info (List[Dict]): 各入力ファイルのグローバルな時間オフセット情報を含むリスト。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"エクスポートを開始します。出力ディレクトリ: {output_dir}")

    # 1. セッション全体のテキストファイルを出力
    session_txt_file = output_dir / f"{session_name}.txt"
    _write_txt_output(timeline, session_txt_file)

    # 2. 各入力ファイルごとのSRT字幕ファイルを出力
    for file_info in source_files_info:
        original_file_stem = file_info["original_path"].stem
        file_global_start_sec = file_info["global_start_sec"]
        file_global_end_sec = file_info["global_end_sec"]

        # このファイルに属するセグメントをフィルタリングし、ローカルタイムスタンプに変換
        local_segments_for_file = []
        for segment in timeline:
            if file_global_start_sec <= segment["start"] < file_global_end_sec:
                local_start = segment["start"] - file_global_start_sec
                local_end = segment["end"] - file_global_start_sec
                local_segments_for_file.append(
                    {
                        "start": local_start,
                        "end": local_end,
                        "text": segment["text"],
                        "speaker": segment["speaker"],
                    }
                )

        if local_segments_for_file:
            file_srt_file = output_dir / f"{original_file_stem}.srt"
            _write_srt_output(local_segments_for_file, file_srt_file)
        else:
            logger.warning(
                f"ファイル {original_file_stem} に対応するセグメントが" + "\n" +
                "見つかりませんでした。SRTファイルは生成されません。" + "\n")

    logger.info("エクスポートが完了しました。\n")
