#!/usr/bin/env python3
"""
改良版話者分離文字起こしシステム
話者分離精度を大幅に向上させた版

主要改善点:
1. 全体ファイルでの話者分離を優先
2. 階層的話者クラスタリング
3. 話者エンベディング活用
4. チャンク間話者一致性向上
"""

import os
import logging
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from pyannote.audio import Pipeline
from pyannote.core import Segment, Annotation
from pyannote.audio.core.inference import Inference
from ladit_pipe.utils.ffmpeg_wrapper import get_duration_sec
from tqdm import tqdm

from sklearn.cluster import AgglomerativeClustering
from pyannote.audio.pipelines.utils.hook import ProgressHook
import numpy as np

# 設定定数
SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
SUPPORTED_VIDEO_FORMATS = {".mkv", ".mp4", ".mov", ".avi", ".webm"}
TARGET_SAMPLE_RATE = 16000
CHUNK_LENGTH_MS = 30000  # 30秒（文字起こし用）
DIARIZATION_CHUNK_LENGTH = 120000  # 2分（話者分離用・長く設定）
MAX_DIARIZATION_LENGTH = 600000  # 10分（話者分離の最大長）
MIN_SILENCE_LEN = 500
SILENCE_THRESH = -40

# ロガー設定
logger = logging.getLogger(__name__)


def perform_chunk_diarization(
    wav_file: Path,
    hf_token: Optional[str],
    device: str,
    min_speakers: int,
    max_speakers: int,
) -> Annotation:
    """
    単一チャンクでの話者分離を実行し、コサイン距離に基づいて話者数を動的に決定します。
    """
    logger.info(f"単一チャンクでの話者分離パイプラインを開始: {wav_file.name}")

    if hf_token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
    else:
        logger.error(
            "Hugging Faceトークンが設定されていません。話者分離パイプラインは実行できません。"
        )
        raise ValueError("Hugging Face token is not set.")

    try:
        # Pyannote Pipelineの初期化
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
        )
        pipeline.to(torch.device(device))
        logger.info("Pyannote話者分離パイプラインをロードしました。")

        # tqdmプログレスバーと連携するフックを作成
        with ProgressHook() as hook:
            logger.info("Pyannoteによる初期話者分離を開始します...")
            diarization_result = pipeline(
                str(wav_file),
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                hook=hook,
            )

        logger.info("Pyannoteによる初期話者分離が完了しました。")

        # 【！】ここが、最後の、そして、最も重要な、アルゴリズムの再構築

        # ステップ1：全ての発話セグメント（turn）から、直接、声紋を抽出する
        embedding_model = Inference("pyannote/embedding")
        embedding_model.to(torch.device(device))
        logger.info("話者エンベディングモデルをロードしました。")

        embeddings = []
        segments = []
        waveform_full, sr_full = torchaudio.load(
            str(wav_file)
        )  # ファイル全体を一度だけロード

        for turn, _, _ in tqdm(
            diarization_result.itertracks(yield_label=True),
            desc="Extracting embeddings from turns",
        ):
            try:
                if turn.duration <= 0:
                    logger.warning(
                        "Skipping turn with non-positive duration: %ss",
                        turn.duration,
                    )
                    continue

                # ターンに対応する波形を切り出す
                start_frame = int(turn.start * sr_full)
                end_frame = int(turn.end * sr_full)

                # 範囲チェック
                if (
                    start_frame >= waveform_full.shape[1]
                    or end_frame > waveform_full.shape[1]
                    or start_frame >= end_frame
                ):
                    logger.warning(
                        "Invalid frame range: start=%s, end=%s, len=%s",
                        start_frame,
                        end_frame,
                        waveform_full.shape[1],
                    )
                    continue

                waveform_turn = waveform_full[:, start_frame:end_frame]

                if waveform_turn.shape[0] > 1:
                    waveform_turn = torch.mean(
                        waveform_turn, dim=0, keepdim=True
                    )

                if sr_full != TARGET_SAMPLE_RATE:
                    resampler = torchaudio.transforms.Resample(
                        sr_full, TARGET_SAMPLE_RATE
                    ).to(device)
                    waveform_turn = resampler(waveform_turn)

                # エンベディングを抽出
                embedding = embedding_model(
                    {
                        "waveform": waveform_turn.to(device),
                        "sample_rate": TARGET_SAMPLE_RATE,
                    }
                )

                if hasattr(embedding, "data") and isinstance(
                    embedding.data, np.ndarray
                ):
                    if (
                        embedding.data.ndim == 2
                        and embedding.data.shape[0] > 0
                    ):
                        avg_embedding = np.mean(embedding.data, axis=0)
                    elif embedding.data.ndim == 1:
                        avg_embedding = embedding.data
                    else:
                        logger.warning(
                            "Skipping turn due to unexpected "
                            "embedding shape: %s",
                            embedding.data.shape,
                        )
                        continue
                else:
                    logger.warning(
                        "Skipping turn due to non-array embedding data: %s",
                        type(embedding.data),
                    )
                    continue

                embeddings.append(avg_embedding)
                segments.append(turn)

            except Exception as e:
                logger.error(
                    f"エンベディング抽出中にエラー: {e}", exc_info=True
                )
                continue

        if not embeddings:
            logger.warning(
                "Embeddings not extracted. Fallback to single speaker."
            )
            duration = get_duration_sec(wav_file)
            fallback_diarization = Annotation(uri=wav_file.name)
            fallback_diarization[Segment(0, duration)] = "SPEAKER_00"
            return fallback_diarization

        embeddings = np.array(embeddings)

        # ステップ2：抽出された「発話ごとの声紋」を、クラスタリングする
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.7,  # ← ここを修正！
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(embeddings)
        n_final_speakers = len(set(labels))
        logger.info(
            "クラスタリングの結果、%s人の最終話者を検出しました。",
            n_final_speakers,
        )

        # ステップ3：各発話セグメントに、最終的な話者IDを割り当てる
        final_diarization = Annotation(uri=wav_file.name)
        for i, segment in enumerate(segments):
            final_diarization[segment] = f"SPEAKER_{labels[i]:02d}"

        return final_diarization

    except Exception as e:
        logger.error(
            f"単一チャンクでの話者分離中にエラーが発生しました: {e}",
            exc_info=True,
        )
