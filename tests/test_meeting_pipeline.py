import pytest
from pathlib import Path
import tempfile
import shutil
import argparse
import os
from pydub import AudioSegment
from pydub.generators import Sine

# ladit_pipeのパスをシステムパスに追加
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from ladit_pipe.pipeline import execute_pipeline
from ladit_pipe.presets.meeting import convert_to_wav_with_progress


@pytest.fixture
def dummy_audio_file():
    """
    短いダミーのWAVファイルを作成するpytestフィクスチャ
    """
    temp_dir = tempfile.mkdtemp()
    file_path = Path(temp_dir) / "dummy_audio.wav"
    # 2秒間の無音オーディオを生成
    audio = AudioSegment.silent(duration=2000)
    audio.export(file_path, format="wav")
    yield file_path
    shutil.rmtree(temp_dir)


def test_convert_to_wav_with_progress(dummy_audio_file):
    """
    convert_to_wav_with_progress関数がプログレスバー付きでエラーなく実行されるかテスト
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            output_path = convert_to_wav_with_progress(dummy_audio_file, Path(temp_dir))
            assert output_path.exists()
            assert output_path.name == "dummy_audio_converted.wav"
        except Exception as e:
            pytest.fail(f"convert_to_wav_with_progress failed with an exception: {e}")


@pytest.fixture
def test_wav_file():
    """
    ./input/test.wav をテスト用の入力ファイルとして提供するフィクスチャ。
    ファイルが存在しない場合は、ダミーファイルを生成する。
    """
    input_dir = Path("./input")
    input_dir.mkdir(exist_ok=True)
    test_file_path = input_dir / "test.wav"

    if not test_file_path.exists():
        print(
            f"'{test_file_path}' が見つからないため、ダミーの音声ファイルを生成します。"
        )
        # 10秒の音声を作成 (2話者)
        tone1 = (
            Sine(440)
            .to_audio_segment(duration=5000)
            .set_channels(1)
            .set_frame_rate(16000)
        )
        tone2 = (
            Sine(880)
            .to_audio_segment(duration=5000)
            .set_channels(1)
            .set_frame_rate(16000)
        )
        audio = tone1 + tone2
        audio.export(test_file_path, format="wav")

    yield test_file_path


@pytest.fixture
def output_dir():
    """
    テストごとに一時的な出力ディレクトリを作成し、テスト後に削除するフィクスチャ。
    """
    temp_output_dir = Path("./test_output_meeting")
    temp_output_dir.mkdir(exist_ok=True)
    yield temp_output_dir
    shutil.rmtree(temp_output_dir)


def test_meeting_pipeline_e2e(test_wav_file, output_dir):
    """
    ミーティングプリセットのエンドツーエンドテスト。
    ./input/test.wav を処理し、期待される出力が生成されることを確認する。
    """
    os.environ["LADIT_PIPE_TESTING"] = "1"
    try:
        args = argparse.Namespace(
            preset="meeting",
            input=test_wav_file,
            output=output_dir,
            whisper_model="tiny",  # テスト用に小さいモデルを使用
            device="cpu",  # テスト用にCPUを使用
            hf_token=os.environ.get("HF_TOKEN", ""),  # 環境変数からHF_TOKENを取得
            min_speakers=1,
            max_speakers=2,
            diarization_threshold=0.5, # テスト用にデフォルト値を追加
            verbose=True,
        )

        execute_pipeline(args)
    finally:
        del os.environ["LADIT_PIPE_TESTING"]

    session_name = test_wav_file.stem
    expected_files = ["txt", "srt", "vtt", "csv", "json"]
    for ext in expected_files:
        f = output_dir / f"{session_name}.{ext}"
        assert f.exists(), f"Expected output file {f} not found."
        assert f.stat().st_size > 0, f"Expected output file {f} is empty."
