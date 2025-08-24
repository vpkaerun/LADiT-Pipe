import logging

import torch
from pyannote.audio import Pipeline
from pyannote.core import Segment, Annotation

# ロガー設定
logger = logging.getLogger(__name__)


def diarize_chunk(
    chunk_file: str,
    diarization_pipeline: Pipeline,
    min_speakers: int,
    max_speakers: int,
) -> Annotation:
    """
    指定された音声チャンクに対して話者分離を実行します。
    pyannote.audioパイプラインを使用し、話者数の制約を適用します。

    Args:
        chunk_file (str): 話者分離を行う音声チャンクのファイルパス。
        diarization_pipeline (Pipeline): 初期化されたpyannote.audioのPipelineオブジェクト。
        min_speakers (int): 話者分離における最小話者数。
        max_speakers (int): 話者分離における最大話者数。

    Returns:
        Annotation: 話者分離結果を含むpyannote.core.Annotationオブジェクト。
                    エラー発生時は空のAnnotationオブジェクトを返します。
    """
    try:
        # GPU メモリ管理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 話者数制約付きで実行
        diarization_params = {
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
        }

        logger.info(
            f"話者分離実行: {chunk_file} (min={min_speakers}, max={max_speakers})"
        )
        diarization = diarization_pipeline(chunk_file, **diarization_params)

        # 話者ラベルの一貫性を向上させる後処理
        diarization = _improve_speaker_consistency(diarization)

        torch.cuda.empty_cache()
        return diarization
    except Exception as e:
        logger.error(f"話者分離エラー {chunk_file}: {e}")
        return Annotation()


def _improve_speaker_consistency(diarization: Annotation) -> Annotation:
    """
    話者分離結果のAnnotationオブジェクトにおいて、話者ラベルの一貫性を改善します。
    非常に短いセグメントを隣接セグメントに統合し、同一話者の近接セグメントを結合します。

    Args:
        diarization (Annotation): 改善するpyannote.core.Annotationオブジェクト。

    Returns:
        Annotation: 改善された話者分離結果を含むAnnotationオブジェクト。
    """
    if len(diarization) == 0:
        return diarization

    # より短いセグメントも統合対象に（話者変化を細かく検出するため）
    MIN_SEGMENT_DURATION = 0.3  # 0.3秒未満のセグメントを統合

    improved = Annotation()
    segments = list(diarization.itertracks(yield_label=True))

    if not segments:
        return diarization

    # セグメントを時間順でソート
    segments.sort(key=lambda x: x[0].start)

    # 第1段階：非常に短いセグメントを隣接セグメントに統合
    merged_segments = []
    i = 0
    while i < len(segments):
        turn, _, speaker = segments[i]

        # 短いセグメントの場合、隣接セグメントと統合を検討
        if turn.duration < MIN_SEGMENT_DURATION and len(segments) > 1:
            merged = False

            # 前のセグメントをチェック
            if i > 0:
                prev_turn, _, prev_speaker = (
                    merged_segments[-1] if merged_segments else segments[i - 1]
                )
                gap_before = turn.start - prev_turn.end
                if gap_before < 0.5:  # 0.5秒以内の間隔
                    # 前のセグメントに統合
                    if merged_segments:
                        merged_segments[-1] = (
                            Segment(prev_turn.start, turn.end),
                            None,
                            prev_speaker,
                        )
                    else:
                        # 最初のセグメントが短い場合、次のセグメントと結合するためにスキップ
                        pass
                    merged = True

            # 次のセグメントをチェック（前と統合されていない場合）
            if not merged and i < len(segments) - 1:
                next_turn, _, next_speaker = segments[i + 1]
                gap_after = next_turn.start - turn.end
                if gap_after < 0.5:  # 0.5秒以内の間隔
                    # 現在のセグメントを次の話者に変更
                    segments[i] = (turn, _, next_speaker)

            if merged:
                i += 1
                continue

        merged_segments.append(segments[i])
        i += 1

    # 第2段階：同一話者の近接セグメントを統合
    final_segments = []
    for i, (turn, _, speaker) in enumerate(merged_segments):
        if (
            final_segments
            and final_segments[-1][2] == speaker  # 同一話者
            and turn.start - final_segments[-1][0].end < 1.0  # 1秒以内の間隔
        ):
            # 前のセグメントと統合
            prev_turn = final_segments[-1][0]
            final_segments[-1] = (Segment(prev_turn.start, turn.end), None, speaker)
        else:
            final_segments.append((turn, _, speaker))

    # Annotationオブジェクトに変換
    for turn, _, speaker in final_segments:
        improved[turn] = speaker

    return improved
