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
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import shutil
import tempfile
import gc

import torch
import torchaudio
import whisper
from pyannote.audio import Pipeline
from pyannote.core import Segment, Annotation
from pyannote.audio.core.inference import Inference
from ladit_pipe.utils.ffmpeg_wrapper import get_duration_sec
from pydub import AudioSegment
from pydub.silence import split_on_silence
from tqdm import tqdm
import librosa

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pyannote.audio.pipelines.utils.hook import ProgressHook

# 設定定数
SUPPORTED_AUDIO_FORMATS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}
SUPPORTED_VIDEO_FORMATS = {'.mkv', '.mp4', '.mov', '.avi', '.webm'}
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
        os.environ['HUGGINGFACE_HUB_TOKEN'] = hf_token
    else:
        logger.error("Hugging Faceトークンが設定されていません。話者分離パイプラインは実行できません。")
        raise ValueError("Hugging Face token is not set.")

    try:
        # Pyannote Pipelineの初期化
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
        pipeline.to(torch.device(device))
        logger.info("Pyannote話者分離パイプラインをロードしました。")

        # tqdmプログレスバーと連携するフックを作成
        with ProgressHook() as hook:
            logger.info("Pyannoteによる初期話者分離を開始します...")
            diarization_result = pipeline(
                str(wav_file),
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                hook=hook
            )

        logger.info("Pyannoteによる初期話者分離が完了しました。")

        # 話者エンベディングの抽出
        embedding_model = Inference("pyannote/embedding")
        embedding_model.to(torch.device(device))
        logger.info("話者エンベディングモデルをロードしました。")

        embeddings = []
        segments = []
        for turn, _, speaker_label in diarization_result.itertracks(yield_label=True):
            logger.debug(f"Processing turn: start={turn.start}, duration={turn.duration}, speaker={speaker_label}")
            
            # 各話者ターンの音声セグメントからエンベディングを抽出
            try:
                if turn.duration <= 0:
                    logger.warning(f"Skipping turn with non-positive duration: {turn.duration}s")
                    continue

                current_sr = TARGET_SAMPLE_RATE 
                num_frames = int(turn.duration * current_sr)
                if num_frames <= 0:
                    logger.warning(f"Skipping turn with zero or negative num_frames after calculation: {num_frames}")
                    continue

                waveform, sr = torchaudio.load(
                    str(wav_file), 
                    frame_offset=int(turn.start * current_sr), 
                    num_frames=num_frames
                )
                logger.debug(f"Loaded waveform shape: {waveform.shape}, sample_rate: {sr}")

                if waveform.shape[0] > 1: 
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                if sr != TARGET_SAMPLE_RATE:
                    logger.debug(f"Resampling from {sr} to {TARGET_SAMPLE_RATE}")
                    resampler = torchaudio.transforms.Resample(sr, TARGET_SAMPLE_RATE).to(device)
                    waveform = resampler(waveform)

                try:
                    embedding = embedding_model({"waveform": waveform.to(device), "sample_rate": TARGET_SAMPLE_RATE})
                    logger.debug(f"Extracted embedding: {embedding.data}")
                except Exception as e:
                    logger.error(f"embedding_model呼び出し中にエラーが発生しました (Turn: {turn.start}-{turn.end}, Speaker: {speaker_label}): {e}", exc_info=True)
                    continue
                
                # エンベディングが全てゼロでないことを確認
                if isinstance(embedding.data, torch.Tensor):
                    embedding_np = embedding.data.cpu().numpy().flatten()
                else:
                    embedding_np = embedding.data.flatten()
                
                if np.all(embedding_np == 0):
                    logger.warning(f"Skipping turn due to all-zero embedding (Turn: {turn.start}-{turn.end}, Speaker: SPEAKER_00)")
                    continue

                embeddings.append(embedding_np)
                logger.debug(f"Current embeddings list size: {len(embeddings)}")
                segments.append(turn)
            except Exception as e:
                logger.error(f"エンベディング抽出中にエラーが発生しました (Turn: {turn.start}-{turn.end}, Speaker: {speaker_label}): {e}", exc_info=True)
                continue
        
        if not embeddings:
            logger.debug(f"Final embeddings list size before check: {len(embeddings)}")
            logger.warning("エンベディングが抽出されませんでした。単一話者としてフォールバックします。")
            # 不安定なdiarization_resultを返す代わりに、ファイル全体を単一話者とする安定したAnnotationを生成
            duration = get_duration_sec(wav_file)
            fallback_diarization = Annotation(uri=wav_file.name)
            fallback_diarization[Segment(0, duration)] = "SPEAKER_00"
            return fallback_diarization

        embeddings = np.array(embeddings)
        logger.info(f"抽出されたエンベディング数: {len(embeddings)}")

        final_diarization = Annotation(uri=wav_file.name)

    except Exception as e:
        logger.error(f"単一チャンクでの話者分離中にエラーが発生しました: {e}", exc_info=True)

def find_audio_files(input_path: Path) -> List[Path]:
    """対応音声・動画ファイルを検索"""
    files = []
    supported_extensions = SUPPORTED_AUDIO_FORMATS | SUPPORTED_VIDEO_FORMATS
    
    if input_path.is_file():
        if input_path.suffix.lower() in supported_extensions:
            files.append(input_path)
    elif input_path.is_dir():
        for ext in supported_extensions:
            files.extend(input_path.rglob(f"*{ext}"))
    
    return sorted(files)

def main():
    parser = argparse.ArgumentParser(description="改良版高性能話者分離文字起こしシステム")
    parser.add_argument("input", type=Path, help="入力ファイルまたはディレクトリ")
    parser.add_argument("-o", "--output", type=Path, default="output", help="出力ディレクトリ")
    parser.add_argument("--whisper-model", default="large-v3", help="Whisperモデル名")
    parser.add_argument("--hf-token", help="Hugging Face トークン")
    parser.add_argument("--device", default="cuda", help="デバイス (cuda/cpu)")
    parser.add_argument("--resume", action="store_true", help="中断された処理を再開")
    parser.add_argument("--min-speakers", type=int, default=1, help="最小話者数 (デフォルト: 1)")
    parser.add_argument("--max-speakers", type=int, default=3, help="最大話者数 (デフォルト: 3)")
    
    args = parser.parse_args()
    
    # 出力ディレクトリ作成
    args.output.mkdir(parents=True, exist_ok=True)
    
    # 一時ディレクトリ作成
    temp_dir = Path(tempfile.mkdtemp(prefix="improved_transcription_"))
    
    # 状態管理
    state_file = args.output / "processing_state.json"
    state = ProcessingState(state_file)
    
    # 対応ファイル検索
    input_files = find_audio_files(args.input)
    if not input_files:
        print(f"対応ファイルが見つかりません: {args.input}")
        sys.exit(1)
    
    print(f"処理対象: {len(input_files)}個のファイル")
    print("=== 改良版話者分離システムの特徴 ===")
    print("1. 全体ファイルでの話者分離を優先")
    print("2. 長いファイルは大きなチャンクで分離後、話者エンベディングで統合")
    print("3. 文字起こしは小さなチャンクで高精度処理")
    print("4. 階層的クラスタリングによる話者統合")
    print("=====================================")
    
    try:
        # プロセッサ初期化
        processor = ImprovedAudioProcessor(temp_dir, args.device)
        processor.initialize_models(
            args.whisper_model, 
            args.hf_token,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers
        )
        
        # ファイル処理
        success_count = 0
        for input_file in input_files:
            if processor.process_file(input_file, args.output, state):
                success_count += 1
        
        print(f"\n処理完了: {success_count}/{len(input_files)} ファイル成功")
        
    except KeyboardInterrupt:
        print("\n処理が中断されました。--resume オプションで再開できます。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        sys.exit(1)
    finally:
        # 一時ディレクトリ削除
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
