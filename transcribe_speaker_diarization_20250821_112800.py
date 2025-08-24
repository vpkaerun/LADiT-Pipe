#!/usr/bin/env python3
"""
高性能話者分離文字起こしシステム
RTX 4060 Laptop (8GB VRAM) 最適化版

機能:
- 複数形式の音声・動画ファイル対応
- インテリジェントなチャンク分割
- GPU最適化された話者分離・文字起こし
- リジューム機能付き耐障害性
- 自動ディスク管理
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
from pydub import AudioSegment
from pydub.silence import split_on_silence
from tqdm import tqdm
import numpy as np

# 設定定数
SUPPORTED_AUDIO_FORMATS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}
SUPPORTED_VIDEO_FORMATS = {'.mkv', '.mp4', '.mov', '.avi', '.webm'}
TARGET_SAMPLE_RATE = 16000
CHUNK_LENGTH_MS = 30000  # 30秒
MAX_CHUNK_LENGTH_MS = 60000  # 60秒
MIN_SILENCE_LEN = 500  # 0.5秒
SILENCE_THRESH = -40  # dB

class ProcessingState:
    """処理状態管理クラス"""
    
    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.state = self.load_state()
    
    def load_state(self) -> Dict:
        """状態ファイルから処理状態を読み込み"""
        if self.state_file.exists():
            with open(self.state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def save_state(self):
        """処理状態を保存"""
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)
    
    def is_file_completed(self, file_path: str) -> bool:
        """ファイル処理が完了しているかチェック"""
        return self.state.get(file_path, {}).get('completed', False)
    
    def mark_file_completed(self, file_path: str):
        """ファイル処理完了をマーク"""
        if file_path not in self.state:
            self.state[file_path] = {}
        self.state[file_path]['completed'] = True
        self.state[file_path]['completed_at'] = datetime.now().isoformat()
        self.save_state()
    
    def get_completed_chunks(self, file_path: str) -> List[int]:
        """完了済みチャンク番号のリストを取得"""
        return self.state.get(file_path, {}).get('completed_chunks', [])
    
    def mark_chunk_completed(self, file_path: str, chunk_idx: int):
        """チャンク処理完了をマーク"""
        if file_path not in self.state:
            self.state[file_path] = {}
        if 'completed_chunks' not in self.state[file_path]:
            self.state[file_path]['completed_chunks'] = []
        
        if chunk_idx not in self.state[file_path]['completed_chunks']:
            self.state[file_path]['completed_chunks'].append(chunk_idx)
        self.save_state()

class AudioProcessor:
    """音声処理クラス"""
    
    def __init__(self, temp_dir: Path, device: str = "cuda"):
        self.temp_dir = temp_dir
        self.device = device
        self.whisper_model = None
        self.diarization_pipeline = None
        
        # ロギング設定
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('transcription.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize_models(self, whisper_model_name: str = "large-v3", 
                         hf_token: Optional[str] = None,
                         min_speakers: int = 1,
                         max_speakers: int = 3):
        """モデルの初期化（一度だけ実行）"""
        self.logger.info("モデルを初期化中...")
        
        # GPU メモリクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Whisper モデル初期化
        self.logger.info(f"Whisper {whisper_model_name} モデルをロード中...")
        self.whisper_model = whisper.load_model(whisper_model_name, device=self.device)
        
        # 話者分離パイプライン初期化
        self.logger.info("Pyannote 話者分離パイプラインをロード中...")
        if hf_token:
            os.environ['HUGGINGFACE_HUB_TOKEN'] = hf_token
        
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        
        if torch.cuda.is_available():
            self.diarization_pipeline.to(torch.device(self.device))
        
        # 話者分離パラメータの最適化設定（より敏感に）
        self.logger.info(f"話者分離パラメータ設定: min_speakers={min_speakers}, max_speakers={max_speakers}")
        
        # より敏感な話者変化検出設定
        if hasattr(self.diarization_pipeline, '_segmentation'):
            # より敏感な設定：話者変化をより細かく検出
            self.diarization_pipeline._segmentation.onset = 0.3      # デフォルト0.5 -> 0.3
            self.diarization_pipeline._segmentation.offset = 0.3     # デフォルト0.5 -> 0.3
            self.logger.info("セグメンテーション感度を高感度に調整しました (0.3/0.3)")
        
        # クラスタリングパラメータの調整
        if hasattr(self.diarization_pipeline, '_clustering'):
            # より厳密なクラスタリング
            if hasattr(self.diarization_pipeline._clustering, 'threshold'):
                self.diarization_pipeline._clustering.threshold = 0.7  # より厳密に
                self.logger.info("クラスタリング閾値を調整しました (0.7)")
        
        # 話者数の制約設定
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        
        self.logger.info("モデル初期化完了")
    
    def convert_to_wav(self, input_file: Path) -> Path:
        """音声ファイルを16kHz モノラル WAV に変換"""
        output_file = self.temp_dir / f"{input_file.stem}_converted.wav"
        
        if output_file.exists():
            return output_file
        
        cmd = [
            'ffmpeg', '-i', str(input_file),
            '-ar', str(TARGET_SAMPLE_RATE),
            '-ac', '1',
            '-c:a', 'pcm_s16le',
            '-y',
            str(output_file)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            self.logger.info(f"変換完了: {input_file.name} -> {output_file.name}")
            return output_file
        except subprocess.CalledProcessError as e:
            self.logger.error(f"変換エラー {input_file}: {e.stderr.decode()}")
            raise
    
    def create_intelligent_chunks(self, wav_file: Path) -> List[Tuple[Path, float, float]]:
        """インテリジェントなチャンク分割"""
        self.logger.info(f"チャンク分割開始: {wav_file.name}")
        
        # pydub で音声読み込み
        audio = AudioSegment.from_wav(wav_file)
        total_duration = len(audio)
        
        # VAD による粗い分割
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
                if chunk_duration > MAX_CHUNK_LENGTH_MS:
                    sub_chunks = self._split_by_time(chunk, CHUNK_LENGTH_MS)
                    for j, sub_chunk in enumerate(sub_chunks):
                        sub_chunk_file = self.temp_dir / f"{wav_file.stem}_chunk_{i}_{j}.wav"
                        sub_chunk.export(sub_chunk_file, format="wav")
                        
                        start_time = current_time
                        end_time = current_time + len(sub_chunk) / 1000.0
                        chunks_info.append((sub_chunk_file, start_time, end_time))
                        current_time = end_time
                else:
                    chunk_file = self.temp_dir / f"{wav_file.stem}_chunk_{i}.wav"
                    chunk.export(chunk_file, format="wav")
                    
                    start_time = current_time
                    end_time = current_time + chunk_duration / 1000.0
                    chunks_info.append((chunk_file, start_time, end_time))
                    current_time = end_time
        else:
            # 短いファイルはそのまま使用
            chunks_info.append((wav_file, 0.0, total_duration / 1000.0))
        
        self.logger.info(f"チャンク分割完了: {len(chunks_info)}個のチャンク")
        return chunks_info
    
    def _split_by_time(self, audio: AudioSegment, chunk_length_ms: int) -> List[AudioSegment]:
        """時間ベースでオーディオを分割"""
        chunks = []
        for i in range(0, len(audio), chunk_length_ms):
            chunk = audio[i:i + chunk_length_ms]
            chunks.append(chunk)
        return chunks
    
    def diarize_chunk(self, chunk_file: Path) -> Annotation:
        """チャンクの話者分離（改良版）"""
        try:
            # GPU メモリ管理
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 話者数制約付きで実行（重要な改善点）
            diarization_params = {
                "min_speakers": self.min_speakers,
                "max_speakers": self.max_speakers
            }
            
            self.logger.info(f"話者分離実行: {chunk_file.name} (min={self.min_speakers}, max={self.max_speakers})")
            diarization = self.diarization_pipeline(str(chunk_file), **diarization_params)
            
            # 話者ラベルの一貫性を向上させる後処理
            diarization = self._improve_speaker_consistency(diarization)
            
            return diarization
        except Exception as e:
            self.logger.error(f"話者分離エラー {chunk_file}: {e}")
            return Annotation()
    
    def _improve_speaker_consistency(self, diarization: Annotation) -> Annotation:
        """話者ラベルの一貫性改善（より厳密版）"""
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
                    prev_turn, _, prev_speaker = segments[i-1]
                    gap_before = turn.start - prev_turn.end
                    if gap_before < 0.5:  # 0.5秒以内の間隔
                        # 前のセグメントに統合
                        merged_segments[-1] = (
                            Segment(prev_turn.start, turn.end),
                            None,
                            prev_speaker
                        )
                        merged = True
                
                # 次のセグメントをチェック（前と統合されていない場合）
                if not merged and i < len(segments) - 1:
                    next_turn, _, next_speaker = segments[i+1]
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
            if (final_segments and 
                final_segments[-1][2] == speaker and  # 同一話者
                turn.start - final_segments[-1][0].end < 1.0):  # 1秒以内の間隔
                # 前のセグメントと統合
                prev_turn = final_segments[-1][0]
                final_segments[-1] = (
                    Segment(prev_turn.start, turn.end),
                    None,
                    speaker
                )
            else:
                final_segments.append((turn, _, speaker))
        
        # Annotationオブジェクトに変換
        for turn, _, speaker in final_segments:
            improved[turn] = speaker
        
        return improved
    
    def transcribe_chunk(self, chunk_file: Path) -> Dict:
        """チャンクの文字起こし（繰り返し抑制版）"""
        try:
            # GPU メモリ管理
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Whisperのオプション設定（繰り返し抑制・高精度化）
            options = {
                "language": "ja",  # 日本語指定
                "task": "transcribe",
                "fp16": True,  # 高速化
                "temperature": 0.0,  # 決定的な出力
                "best_of": 1,  # 高速化
                "beam_size": 1,  # 高速化
                "patience": 1.0,
                # 繰り返し抑制の強化
                "no_repeat_ngram_size": 3,      # 3-gramの繰り返しを防止
                "repetition_penalty": 1.2,      # 繰り返しペナルティ
                "suppress_tokens": [-1],         # 不要なトークンを抑制
                "compression_ratio_threshold": 2.4,  # 圧縮率による品質チェック
                "logprob_threshold": -1.0,       # 低確率出力の抑制
                "condition_on_previous_text": False,  # 前のテキストに依存しない
            }
            
            result = self.whisper_model.transcribe(str(chunk_file), **options)
            
            # 後処理：繰り返しパターンの除去
            result = self._post_process_transcription(result)
            
            return result
        except Exception as e:
            self.logger.error(f"文字起こしエラー {chunk_file}: {e}")
            return {"segments": [], "text": ""}
    
    def _post_process_transcription(self, result: Dict) -> Dict:
        """転写結果の後処理（繰り返し除去）"""
        if "segments" not in result:
            return result
        
        processed_segments = []
        
        for segment in result["segments"]:
            text = segment.get("text", "").strip()
            
            # 繰り返しパターンを除去
            cleaned_text = self._remove_repetitions(text)
            
            # 空文字や意味のないセグメントをスキップ
            if len(cleaned_text) > 0 and not self._is_meaningless_segment(cleaned_text):
                segment["text"] = cleaned_text
                processed_segments.append(segment)
        
        result["segments"] = processed_segments
        
        # 全体テキストも更新
        result["text"] = " ".join([seg["text"] for seg in processed_segments])
        
        return result
    
    def _remove_repetitions(self, text: str) -> str:
        """繰り返しパターンを除去"""
        import re
        
        # 1. 同一単語の連続繰り返し除去（例：「誠に誠に誠に...」→「誠に」）
        # 3回以上の繰り返しを1回に
        text = re.sub(r'(\S+?)(\1){2,}', r'\1', text)
        
        # 2. 短いフレーズの繰り返し除去（例：「そうですねそうですね」→「そうですね」）
        words = text.split()
        if len(words) > 6:  # 十分な長さがある場合のみ
            # 連続する同一フレーズ（2-3語）を検出・除去
            i = 0
            cleaned_words = []
            while i < len(words):
                # 2語フレーズの繰り返しチェック
                if i + 3 < len(words) and words[i:i+2] == words[i+2:i+4]:
                    cleaned_words.extend(words[i:i+2])
                    # 同じパターンをスキップ
                    j = i + 4
                    while j + 1 < len(words) and words[i:i+2] == words[j:j+2]:
                        j += 2
                    i = j
                # 3語フレーズの繰り返しチェック
                elif i + 5 < len(words) and words[i:i+3] == words[i+3:i+6]:
                    cleaned_words.extend(words[i:i+3])
                    # 同じパターンをスキップ
                    j = i + 6
                    while j + 2 < len(words) and words[i:i+3] == words[j:j+3]:
                        j += 3
                    i = j
                else:
                    cleaned_words.append(words[i])
                    i += 1
            
            text = " ".join(cleaned_words)
        
        # 3. 「あああ」「うううう」などの連続する同一文字の除去
        text = re.sub(r'(\S)\1{4,}', r'\1', text)
        
        return text.strip()
    
    def _is_meaningless_segment(self, text: str) -> bool:
        """意味のないセグメントかどうか判定"""
        # 非常に短い
        if len(text.strip()) < 2:
            return True
        
        # 単一文字の繰り返しのみ
        if re.match(r'^(\S)\1*
    
    def merge_results(self, chunk_results: List[Tuple[Dict, Annotation, float, float]], 
                     output_base: Path) -> Dict[str, Path]:
        """結果をマージして出力ファイル生成（改良版）"""
        
        merged_segments = []
        all_speakers = set()
        
        # まず全チャンクから話者情報を収集
        for transcription, diarization, chunk_start, chunk_end in chunk_results:
            for turn, _, speaker_label in diarization.itertracks(yield_label=True):
                all_speakers.add(speaker_label)
        
        self.logger.info(f"検出された話者: {sorted(all_speakers)}")
        
        # 話者ラベルの正規化マッピング作成
        speaker_mapping = self._create_speaker_mapping(all_speakers)
        self.logger.info(f"話者マッピング: {speaker_mapping}")
        
        # Whisperのセグメントと話者情報をマージ
        for transcription, diarization, chunk_start, chunk_end in chunk_results:
            for segment in transcription.get("segments", []):
                start_time = chunk_start + segment["start"]
                end_time = chunk_start + segment["end"]
                text = segment["text"].strip()
                
                if not text:
                    continue
                
                # 該当時間の話者を特定（改良版）
                speaker = self._find_best_speaker(
                    diarization, start_time, end_time, chunk_start, speaker_mapping
                )
                
                merged_segments.append({
                    "start": start_time,
                    "end": end_time,
                    "text": text,
                    "speaker": speaker
                })
        
        # 時間順でソート
        merged_segments.sort(key=lambda x: x["start"])
        
        # 同一話者の連続セグメントを統合
        merged_segments = self._merge_consecutive_segments(merged_segments)
        
        # 出力ファイル生成
        output_files = {}
        
        # 安全なファイル名生成（時刻を含むファイル名に対応）
        base_name = output_base.stem
        output_dir = output_base.parent
        
        # テキスト形式
        txt_file = output_dir / f"{base_name}_diarized.txt"
        self._write_txt_output(merged_segments, txt_file)
        output_files["txt"] = txt_file
        
        # SRT 形式
        srt_file = output_dir / f"{base_name}.srt"
        self._write_srt_output(merged_segments, srt_file)
        output_files["srt"] = srt_file
        
        # VTT 形式
        vtt_file = output_dir / f"{base_name}.vtt"
        self._write_vtt_output(merged_segments, vtt_file)
        output_files["vtt"] = vtt_file
        
        return output_files
    
    def _create_speaker_mapping(self, all_speakers: set) -> Dict[str, str]:
        """話者ラベルの正規化マッピング作成"""
        mapping = {}
        speaker_counter = 0
        
        # 数値でソートして一貫性を保つ
        sorted_speakers = sorted(all_speakers)
        
        for original_label in sorted_speakers:
            mapping[original_label] = f"SPEAKER_{speaker_counter}"
            speaker_counter += 1
        
        return mapping
    
    def _find_best_speaker(self, diarization: Annotation, start_time: float, 
                          end_time: float, chunk_start: float, 
                          speaker_mapping: Dict[str, str]) -> str:
        """最適な話者を特定（改良版）"""
        
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
        min_distance = float('inf')
        
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
    
    def _merge_consecutive_segments(self, segments: List[Dict]) -> List[Dict]:
        """同一話者の連続セグメントを統合（より厳密版）"""
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
    
    def _write_txt_output(self, segments: List[Dict], output_file: Path):
        """テキスト形式で出力"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for segment in segments:
                start_str = self._format_timestamp(segment["start"])
                end_str = self._format_timestamp(segment["end"])
                f.write(f"[{start_str} - {end_str}] {segment['speaker']}: {segment['text']}\n")
    
    def _write_srt_output(self, segments: List[Dict], output_file: Path):
        """SRT 字幕形式で出力"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, 1):
                start_str = self._format_srt_timestamp(segment["start"])
                end_str = self._format_srt_timestamp(segment["end"])
                f.write(f"{i}\n")
                f.write(f"{start_str} --> {end_str}\n")
                f.write(f"<v {segment['speaker']}>{segment['text']}\n\n")
    
    def _write_vtt_output(self, segments: List[Dict], output_file: Path):
        """VTT 字幕形式で出力"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            for segment in segments:
                start_str = self._format_vtt_timestamp(segment["start"])
                end_str = self._format_vtt_timestamp(segment["end"])
                f.write(f"{start_str} --> {end_str}\n")
                f.write(f"<v {segment['speaker']}>{segment['text']}\n\n")
    
    def _format_timestamp(self, seconds: float) -> str:
        """タイムスタンプフォーマット (HH:MM:SS.mmm)"""
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"
    
    def _format_srt_timestamp(self, seconds: float) -> str:
        """SRT タイムスタンプフォーマット"""
        return self._format_timestamp(seconds).replace('.', ',')
    
    def _format_vtt_timestamp(self, seconds: float) -> str:
        """VTT タイムスタンプフォーマット"""
        return self._format_timestamp(seconds)
    
    def process_file(self, input_file: Path, output_dir: Path, state: ProcessingState) -> bool:
        """単一ファイルを処理"""
        file_key = str(input_file)
        
        if state.is_file_completed(file_key):
            self.logger.info(f"スキップ (完了済み): {input_file.name}")
            return True
        
        try:
            self.logger.info(f"処理開始: {input_file.name}")
            
            # 1. WAV変換
            wav_file = self.convert_to_wav(input_file)
            
            # 2. チャンク分割
            chunks_info = self.create_intelligent_chunks(wav_file)
            
            # 3. 各チャンク処理
            chunk_results = []
            completed_chunks = state.get_completed_chunks(file_key)
            
            with tqdm(chunks_info, desc=f"Processing {input_file.name}") as pbar:
                for i, (chunk_file, chunk_start, chunk_end) in enumerate(pbar):
                    if i in completed_chunks:
                        pbar.set_postfix(status="cached")
                        continue
                    
                    pbar.set_postfix(status="diarizing")
                    # 話者分離
                    diarization = self.diarize_chunk(chunk_file)
                    
                    pbar.set_postfix(status="transcribing")
                    # 文字起こし
                    transcription = self.transcribe_chunk(chunk_file)
                    
                    chunk_results.append((transcription, diarization, chunk_start, chunk_end))
                    state.mark_chunk_completed(file_key, i)
                    
                    # GPU メモリクリア
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                    pbar.set_postfix(status="completed")
            
            # 4. 結果マージ・出力
            self.logger.info(f"結果を統合中: {input_file.name}")
            output_base = output_dir / input_file.stem
            output_files = self.merge_results(chunk_results, output_base)
            
            # 5. 中間ファイル削除
            self._cleanup_temp_files(wav_file, chunks_info)
            
            # 6. 完了マーク
            state.mark_file_completed(file_key)
            
            self.logger.info(f"処理完了: {input_file.name}")
            for format_type, file_path in output_files.items():
                self.logger.info(f"  {format_type.upper()}: {file_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"ファイル処理エラー {input_file}: {e}")
            import traceback
            self.logger.error(f"詳細エラー:\n{traceback.format_exc()}")
            return False
    
    def _cleanup_temp_files(self, wav_file: Path, chunks_info: List):
        """中間ファイルのクリーンアップ"""
        try:
            if wav_file.exists() and wav_file.parent == self.temp_dir:
                wav_file.unlink()
            
            for chunk_file, _, _ in chunks_info:
                if chunk_file.exists() and chunk_file.parent == self.temp_dir:
                    chunk_file.unlink()
        except Exception as e:
            self.logger.warning(f"中間ファイル削除警告: {e}")

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
    parser = argparse.ArgumentParser(description="高性能話者分離文字起こしシステム")
    parser.add_argument("input", type=Path, help="入力ファイルまたはディレクトリ")
    parser.add_argument("-o", "--output", type=Path, default="output", help="出力ディレクトリ")
    parser.add_argument("--whisper-model", default="large-v3", help="Whisperモデル名")
    parser.add_argument("--hf-token", help="Hugging Face トークン")
    parser.add_argument("--device", default="cuda", help="デバイス (cuda/cpu)")
    parser.add_argument("--resume", action="store_true", help="中断された処理を再開")
    parser.add_argument("--min-speakers", type=int, default=1, help="最小話者数 (デフォルト: 1)")
    parser.add_argument("--max-speakers", type=int, default=3, help="最大話者数 (デフォルト: 3)")
    parser.add_argument("--sensitive", action="store_true", 
                       help="高感度モード（話者変化をより細かく検出）")
    
    args = parser.parse_args()
    
    # 出力ディレクトリ作成
    args.output.mkdir(parents=True, exist_ok=True)
    
    # 一時ディレクトリ作成
    temp_dir = Path(tempfile.mkdtemp(prefix="transcription_"))
    
    # 状態管理
    state_file = args.output / "processing_state.json"
    state = ProcessingState(state_file)
    
    # 対応ファイル検索
    input_files = find_audio_files(args.input)
    if not input_files:
        print(f"対応ファイルが見つかりません: {args.input}")
        sys.exit(1)
    
    print(f"処理対象: {len(input_files)}個のファイル")
    
    try:
        # プロセッサ初期化
        processor = AudioProcessor(temp_dir, args.device)
        
        # 高感度モードの場合、さらに敏感な設定を適用
        if args.sensitive:
            print("高感度モードが有効です（話者変化をより細かく検出）")
        
        processor.initialize_models(
            args.whisper_model, 
            args.hf_token,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers
        )
        
        # 高感度モード設定の追加調整
        if args.sensitive and hasattr(processor, 'diarization_pipeline'):
            if hasattr(processor.diarization_pipeline, '_segmentation'):
                processor.diarization_pipeline._segmentation.onset = 0.2   # さらに敏感に
                processor.diarization_pipeline._segmentation.offset = 0.2  # さらに敏感に
                print("高感度モード: セグメンテーション閾値を0.2/0.2に設定")
        
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
, text.strip()):
            return True
        
        # 記号のみ
        if re.match(r'^[^\w\s]*
    
    def merge_results(self, chunk_results: List[Tuple[Dict, Annotation, float, float]], 
                     output_base: Path) -> Dict[str, Path]:
        """結果をマージして出力ファイル生成（改良版）"""
        
        merged_segments = []
        all_speakers = set()
        
        # まず全チャンクから話者情報を収集
        for transcription, diarization, chunk_start, chunk_end in chunk_results:
            for turn, _, speaker_label in diarization.itertracks(yield_label=True):
                all_speakers.add(speaker_label)
        
        self.logger.info(f"検出された話者: {sorted(all_speakers)}")
        
        # 話者ラベルの正規化マッピング作成
        speaker_mapping = self._create_speaker_mapping(all_speakers)
        self.logger.info(f"話者マッピング: {speaker_mapping}")
        
        # Whisperのセグメントと話者情報をマージ
        for transcription, diarization, chunk_start, chunk_end in chunk_results:
            for segment in transcription.get("segments", []):
                start_time = chunk_start + segment["start"]
                end_time = chunk_start + segment["end"]
                text = segment["text"].strip()
                
                if not text:
                    continue
                
                # 該当時間の話者を特定（改良版）
                speaker = self._find_best_speaker(
                    diarization, start_time, end_time, chunk_start, speaker_mapping
                )
                
                merged_segments.append({
                    "start": start_time,
                    "end": end_time,
                    "text": text,
                    "speaker": speaker
                })
        
        # 時間順でソート
        merged_segments.sort(key=lambda x: x["start"])
        
        # 同一話者の連続セグメントを統合
        merged_segments = self._merge_consecutive_segments(merged_segments)
        
        # 出力ファイル生成
        output_files = {}
        
        # 安全なファイル名生成（時刻を含むファイル名に対応）
        base_name = output_base.stem
        output_dir = output_base.parent
        
        # テキスト形式
        txt_file = output_dir / f"{base_name}_diarized.txt"
        self._write_txt_output(merged_segments, txt_file)
        output_files["txt"] = txt_file
        
        # SRT 形式
        srt_file = output_dir / f"{base_name}.srt"
        self._write_srt_output(merged_segments, srt_file)
        output_files["srt"] = srt_file
        
        # VTT 形式
        vtt_file = output_dir / f"{base_name}.vtt"
        self._write_vtt_output(merged_segments, vtt_file)
        output_files["vtt"] = vtt_file
        
        return output_files
    
    def _create_speaker_mapping(self, all_speakers: set) -> Dict[str, str]:
        """話者ラベルの正規化マッピング作成"""
        mapping = {}
        speaker_counter = 0
        
        # 数値でソートして一貫性を保つ
        sorted_speakers = sorted(all_speakers)
        
        for original_label in sorted_speakers:
            mapping[original_label] = f"SPEAKER_{speaker_counter}"
            speaker_counter += 1
        
        return mapping
    
    def _find_best_speaker(self, diarization: Annotation, start_time: float, 
                          end_time: float, chunk_start: float, 
                          speaker_mapping: Dict[str, str]) -> str:
        """最適な話者を特定（改良版）"""
        
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
        min_distance = float('inf')
        
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
    
    def _merge_consecutive_segments(self, segments: List[Dict]) -> List[Dict]:
        """同一話者の連続セグメントを統合"""
        if not segments:
            return segments
        
        merged = []
        current_segment = segments[0].copy()
        
        for next_segment in segments[1:]:
            # 同一話者で時間が近接している場合は統合
            time_gap = next_segment["start"] - current_segment["end"]
            same_speaker = current_segment["speaker"] == next_segment["speaker"]
            close_in_time = time_gap < 2.0  # 2秒以内の間隔
            
            if same_speaker and close_in_time:
                # セグメントを統合
                current_segment["end"] = next_segment["end"]
                current_segment["text"] += " " + next_segment["text"]
            else:
                # 現在のセグメントを完了し、次のセグメントを開始
                merged.append(current_segment)
                current_segment = next_segment.copy()
        
        # 最後のセグメントを追加
        merged.append(current_segment)
        
        return merged
    
    def _write_txt_output(self, segments: List[Dict], output_file: Path):
        """テキスト形式で出力"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for segment in segments:
                start_str = self._format_timestamp(segment["start"])
                end_str = self._format_timestamp(segment["end"])
                f.write(f"[{start_str} - {end_str}] {segment['speaker']}: {segment['text']}\n")
    
    def _write_srt_output(self, segments: List[Dict], output_file: Path):
        """SRT 字幕形式で出力"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, 1):
                start_str = self._format_srt_timestamp(segment["start"])
                end_str = self._format_srt_timestamp(segment["end"])
                f.write(f"{i}\n")
                f.write(f"{start_str} --> {end_str}\n")
                f.write(f"<v {segment['speaker']}>{segment['text']}\n\n")
    
    def _write_vtt_output(self, segments: List[Dict], output_file: Path):
        """VTT 字幕形式で出力"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            for segment in segments:
                start_str = self._format_vtt_timestamp(segment["start"])
                end_str = self._format_vtt_timestamp(segment["end"])
                f.write(f"{start_str} --> {end_str}\n")
                f.write(f"<v {segment['speaker']}>{segment['text']}\n\n")
    
    def _format_timestamp(self, seconds: float) -> str:
        """タイムスタンプフォーマット (HH:MM:SS.mmm)"""
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"
    
    def _format_srt_timestamp(self, seconds: float) -> str:
        """SRT タイムスタンプフォーマット"""
        return self._format_timestamp(seconds).replace('.', ',')
    
    def _format_vtt_timestamp(self, seconds: float) -> str:
        """VTT タイムスタンプフォーマット"""
        return self._format_timestamp(seconds)
    
    def process_file(self, input_file: Path, output_dir: Path, state: ProcessingState) -> bool:
        """単一ファイルを処理"""
        file_key = str(input_file)
        
        if state.is_file_completed(file_key):
            self.logger.info(f"スキップ (完了済み): {input_file.name}")
            return True
        
        try:
            self.logger.info(f"処理開始: {input_file.name}")
            
            # 1. WAV変換
            wav_file = self.convert_to_wav(input_file)
            
            # 2. チャンク分割
            chunks_info = self.create_intelligent_chunks(wav_file)
            
            # 3. 各チャンク処理
            chunk_results = []
            completed_chunks = state.get_completed_chunks(file_key)
            
            with tqdm(chunks_info, desc=f"Processing {input_file.name}") as pbar:
                for i, (chunk_file, chunk_start, chunk_end) in enumerate(pbar):
                    if i in completed_chunks:
                        pbar.set_postfix(status="cached")
                        continue
                    
                    pbar.set_postfix(status="diarizing")
                    # 話者分離
                    diarization = self.diarize_chunk(chunk_file)
                    
                    pbar.set_postfix(status="transcribing")
                    # 文字起こし
                    transcription = self.transcribe_chunk(chunk_file)
                    
                    chunk_results.append((transcription, diarization, chunk_start, chunk_end))
                    state.mark_chunk_completed(file_key, i)
                    
                    # GPU メモリクリア
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                    pbar.set_postfix(status="completed")
            
            # 4. 結果マージ・出力
            self.logger.info(f"結果を統合中: {input_file.name}")
            output_base = output_dir / input_file.stem
            output_files = self.merge_results(chunk_results, output_base)
            
            # 5. 中間ファイル削除
            self._cleanup_temp_files(wav_file, chunks_info)
            
            # 6. 完了マーク
            state.mark_file_completed(file_key)
            
            self.logger.info(f"処理完了: {input_file.name}")
            for format_type, file_path in output_files.items():
                self.logger.info(f"  {format_type.upper()}: {file_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"ファイル処理エラー {input_file}: {e}")
            import traceback
            self.logger.error(f"詳細エラー:\n{traceback.format_exc()}")
            return False
    
    def _cleanup_temp_files(self, wav_file: Path, chunks_info: List):
        """中間ファイルのクリーンアップ"""
        try:
            if wav_file.exists() and wav_file.parent == self.temp_dir:
                wav_file.unlink()
            
            for chunk_file, _, _ in chunks_info:
                if chunk_file.exists() and chunk_file.parent == self.temp_dir:
                    chunk_file.unlink()
        except Exception as e:
            self.logger.warning(f"中間ファイル削除警告: {e}")

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
    parser = argparse.ArgumentParser(description="高性能話者分離文字起こしシステム")
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
    temp_dir = Path(tempfile.mkdtemp(prefix="transcription_"))
    
    # 状態管理
    state_file = args.output / "processing_state.json"
    state = ProcessingState(state_file)
    
    # 対応ファイル検索
    input_files = find_audio_files(args.input)
    if not input_files:
        print(f"対応ファイルが見つかりません: {args.input}")
        sys.exit(1)
    
    print(f"処理対象: {len(input_files)}個のファイル")
    
    try:
        # プロセッサ初期化
        processor = AudioProcessor(temp_dir, args.device)
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
, text.strip()):
            return True
        
        return False
    
    def merge_results(self, chunk_results: List[Tuple[Dict, Annotation, float, float]], 
                     output_base: Path) -> Dict[str, Path]:
        """結果をマージして出力ファイル生成（改良版）"""
        
        merged_segments = []
        all_speakers = set()
        
        # まず全チャンクから話者情報を収集
        for transcription, diarization, chunk_start, chunk_end in chunk_results:
            for turn, _, speaker_label in diarization.itertracks(yield_label=True):
                all_speakers.add(speaker_label)
        
        self.logger.info(f"検出された話者: {sorted(all_speakers)}")
        
        # 話者ラベルの正規化マッピング作成
        speaker_mapping = self._create_speaker_mapping(all_speakers)
        self.logger.info(f"話者マッピング: {speaker_mapping}")
        
        # Whisperのセグメントと話者情報をマージ
        for transcription, diarization, chunk_start, chunk_end in chunk_results:
            for segment in transcription.get("segments", []):
                start_time = chunk_start + segment["start"]
                end_time = chunk_start + segment["end"]
                text = segment["text"].strip()
                
                if not text:
                    continue
                
                # 該当時間の話者を特定（改良版）
                speaker = self._find_best_speaker(
                    diarization, start_time, end_time, chunk_start, speaker_mapping
                )
                
                merged_segments.append({
                    "start": start_time,
                    "end": end_time,
                    "text": text,
                    "speaker": speaker
                })
        
        # 時間順でソート
        merged_segments.sort(key=lambda x: x["start"])
        
        # 同一話者の連続セグメントを統合
        merged_segments = self._merge_consecutive_segments(merged_segments)
        
        # 出力ファイル生成
        output_files = {}
        
        # 安全なファイル名生成（時刻を含むファイル名に対応）
        base_name = output_base.stem
        output_dir = output_base.parent
        
        # テキスト形式
        txt_file = output_dir / f"{base_name}_diarized.txt"
        self._write_txt_output(merged_segments, txt_file)
        output_files["txt"] = txt_file
        
        # SRT 形式
        srt_file = output_dir / f"{base_name}.srt"
        self._write_srt_output(merged_segments, srt_file)
        output_files["srt"] = srt_file
        
        # VTT 形式
        vtt_file = output_dir / f"{base_name}.vtt"
        self._write_vtt_output(merged_segments, vtt_file)
        output_files["vtt"] = vtt_file
        
        return output_files
    
    def _create_speaker_mapping(self, all_speakers: set) -> Dict[str, str]:
        """話者ラベルの正規化マッピング作成"""
        mapping = {}
        speaker_counter = 0
        
        # 数値でソートして一貫性を保つ
        sorted_speakers = sorted(all_speakers)
        
        for original_label in sorted_speakers:
            mapping[original_label] = f"SPEAKER_{speaker_counter}"
            speaker_counter += 1
        
        return mapping
    
    def _find_best_speaker(self, diarization: Annotation, start_time: float, 
                          end_time: float, chunk_start: float, 
                          speaker_mapping: Dict[str, str]) -> str:
        """最適な話者を特定（改良版）"""
        
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
        min_distance = float('inf')
        
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
    
    def _merge_consecutive_segments(self, segments: List[Dict]) -> List[Dict]:
        """同一話者の連続セグメントを統合"""
        if not segments:
            return segments
        
        merged = []
        current_segment = segments[0].copy()
        
        for next_segment in segments[1:]:
            # 同一話者で時間が近接している場合は統合
            time_gap = next_segment["start"] - current_segment["end"]
            same_speaker = current_segment["speaker"] == next_segment["speaker"]
            close_in_time = time_gap < 2.0  # 2秒以内の間隔
            
            if same_speaker and close_in_time:
                # セグメントを統合
                current_segment["end"] = next_segment["end"]
                current_segment["text"] += " " + next_segment["text"]
            else:
                # 現在のセグメントを完了し、次のセグメントを開始
                merged.append(current_segment)
                current_segment = next_segment.copy()
        
        # 最後のセグメントを追加
        merged.append(current_segment)
        
        return merged
    
    def _write_txt_output(self, segments: List[Dict], output_file: Path):
        """テキスト形式で出力"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for segment in segments:
                start_str = self._format_timestamp(segment["start"])
                end_str = self._format_timestamp(segment["end"])
                f.write(f"[{start_str} - {end_str}] {segment['speaker']}: {segment['text']}\n")
    
    def _write_srt_output(self, segments: List[Dict], output_file: Path):
        """SRT 字幕形式で出力"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, 1):
                start_str = self._format_srt_timestamp(segment["start"])
                end_str = self._format_srt_timestamp(segment["end"])
                f.write(f"{i}\n")
                f.write(f"{start_str} --> {end_str}\n")
                f.write(f"<v {segment['speaker']}>{segment['text']}\n\n")
    
    def _write_vtt_output(self, segments: List[Dict], output_file: Path):
        """VTT 字幕形式で出力"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            for segment in segments:
                start_str = self._format_vtt_timestamp(segment["start"])
                end_str = self._format_vtt_timestamp(segment["end"])
                f.write(f"{start_str} --> {end_str}\n")
                f.write(f"<v {segment['speaker']}>{segment['text']}\n\n")
    
    def _format_timestamp(self, seconds: float) -> str:
        """タイムスタンプフォーマット (HH:MM:SS.mmm)"""
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"
    
    def _format_srt_timestamp(self, seconds: float) -> str:
        """SRT タイムスタンプフォーマット"""
        return self._format_timestamp(seconds).replace('.', ',')
    
    def _format_vtt_timestamp(self, seconds: float) -> str:
        """VTT タイムスタンプフォーマット"""
        return self._format_timestamp(seconds)
    
    def process_file(self, input_file: Path, output_dir: Path, state: ProcessingState) -> bool:
        """単一ファイルを処理"""
        file_key = str(input_file)
        
        if state.is_file_completed(file_key):
            self.logger.info(f"スキップ (完了済み): {input_file.name}")
            return True
        
        try:
            self.logger.info(f"処理開始: {input_file.name}")
            
            # 1. WAV変換
            wav_file = self.convert_to_wav(input_file)
            
            # 2. チャンク分割
            chunks_info = self.create_intelligent_chunks(wav_file)
            
            # 3. 各チャンク処理
            chunk_results = []
            completed_chunks = state.get_completed_chunks(file_key)
            
            with tqdm(chunks_info, desc=f"Processing {input_file.name}") as pbar:
                for i, (chunk_file, chunk_start, chunk_end) in enumerate(pbar):
                    if i in completed_chunks:
                        pbar.set_postfix(status="cached")
                        continue
                    
                    pbar.set_postfix(status="diarizing")
                    # 話者分離
                    diarization = self.diarize_chunk(chunk_file)
                    
                    pbar.set_postfix(status="transcribing")
                    # 文字起こし
                    transcription = self.transcribe_chunk(chunk_file)
                    
                    chunk_results.append((transcription, diarization, chunk_start, chunk_end))
                    state.mark_chunk_completed(file_key, i)
                    
                    # GPU メモリクリア
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                    pbar.set_postfix(status="completed")
            
            # 4. 結果マージ・出力
            self.logger.info(f"結果を統合中: {input_file.name}")
            output_base = output_dir / input_file.stem
            output_files = self.merge_results(chunk_results, output_base)
            
            # 5. 中間ファイル削除
            self._cleanup_temp_files(wav_file, chunks_info)
            
            # 6. 完了マーク
            state.mark_file_completed(file_key)
            
            self.logger.info(f"処理完了: {input_file.name}")
            for format_type, file_path in output_files.items():
                self.logger.info(f"  {format_type.upper()}: {file_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"ファイル処理エラー {input_file}: {e}")
            import traceback
            self.logger.error(f"詳細エラー:\n{traceback.format_exc()}")
            return False
    
    def _cleanup_temp_files(self, wav_file: Path, chunks_info: List):
        """中間ファイルのクリーンアップ"""
        try:
            if wav_file.exists() and wav_file.parent == self.temp_dir:
                wav_file.unlink()
            
            for chunk_file, _, _ in chunks_info:
                if chunk_file.exists() and chunk_file.parent == self.temp_dir:
                    chunk_file.unlink()
        except Exception as e:
            self.logger.warning(f"中間ファイル削除警告: {e}")

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
    parser = argparse.ArgumentParser(description="高性能話者分離文字起こしシステム")
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
    temp_dir = Path(tempfile.mkdtemp(prefix="transcription_"))
    
    # 状態管理
    state_file = args.output / "processing_state.json"
    state = ProcessingState(state_file)
    
    # 対応ファイル検索
    input_files = find_audio_files(args.input)
    if not input_files:
        print(f"対応ファイルが見つかりません: {args.input}")
        sys.exit(1)
    
    print(f"処理対象: {len(input_files)}個のファイル")
    
    try:
        # プロセッサ初期化
        processor = AudioProcessor(temp_dir, args.device)
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