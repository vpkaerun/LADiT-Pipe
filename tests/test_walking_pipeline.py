import pytest
import argparse
from pathlib import Path
import shutil
import os
from pydub import AudioSegment
from pydub.generators import Sine
import logging

from ladit_pipe.pipeline import execute_pipeline
from ladit_pipe.utils.file_handler import group_files_by_date

# テスト用のダミー音声ファイルが配置されているディレクトリ
TEST_AUDIO_DIR = Path(__file__).parent


@pytest.fixture(scope="module")
def setup_walking_test_files():
    """
    ウォーキングモードのテスト用に、より現実に近いダミー音声ファイルを準備し、
    テスト後にクリーンアップするフィクスチャ。
    """
    test_files = [
        TEST_AUDIO_DIR / "walk_2025-08-30_part1.wav",
        TEST_AUDIO_DIR / "walk_2025-08-30_part2.wav",
        TEST_AUDIO_DIR / "walk_2025-08-31_part1.wav",
    ]

    # 2人の話者をシミュレートする音声を作成
    # 話者A: 440Hz, 話者B: 660Hz
    # [A: 2s] [無音: 0.5s] [B: 2s] [無音: 0.5s] [A: 2s] [B: 2s] のような構成
    speaker_a = (
        Sine(440).to_audio_segment(duration=2000).set_channels(1).set_frame_rate(16000)
    )
    speaker_b = (
        Sine(660).to_audio_segment(duration=2000).set_channels(1).set_frame_rate(16000)
    )
    silence = AudioSegment.silent(duration=500, frame_rate=16000)
    two_speaker_wav = speaker_a + silence + speaker_b + silence + speaker_a + speaker_b

    for f in test_files:
        if not f.exists():
            two_speaker_wav.export(f, format="wav")

    yield  # テスト実行

    # クリーンアップ
    for f in test_files:
        if f.exists():
            f.unlink()


@pytest.fixture
def output_dir():
    """
    テストごとに一時的な出力ディレクトリを作成し、テスト後に削除するフィクスチャ。
    """
    temp_output_dir = Path("./test_output_walking")
    temp_output_dir.mkdir(exist_ok=True)
    yield temp_output_dir
    shutil.rmtree(temp_output_dir)


def test_walking_pipeline_e2e(setup_walking_test_files, output_dir):
    """
    ウォーキングプリセットのエンドツーエンドテスト。
    日付ごとにグループ化されたファイルを処理し、期待される出力が生成されることを確認する。
    """
    os.environ["LADIT_PIPE_TESTING"] = "1"
    try:
        # main関数に渡す引数を準備
        args = argparse.Namespace(
            preset="walking",
            input=TEST_AUDIO_DIR,  # テスト用のダミー音声ファイルがあるディレクトリ
            output=output_dir,
            whisper_model="tiny",  # テスト用に小さいモデルを使用
            device="cpu",  # テスト用にCPUを使用
            hf_token=os.environ.get("HF_TOKEN", ""),  # 環境変数からHF_TOKENを取得
            min_speakers=1,
            max_speakers=2,
            diarization_threshold=0.6, # テスト用にデフォルト値を設定
            verbose=True,  # デバッグログを有効にする
        )

        # main関数を実行
        execute_pipeline(args)
    finally:
        del os.environ["LADIT_PIPE_TESTING"]

    # 出力ディレクトリに期待されるファイルが生成されたかを確認
    # 2025-08-30 と 2025-08-31 の2つの日付グループが処理されるはず
    # export.pyは5種類のファイルを出力するため、すべてをチェックする
    expected_output_files_30 = [
        output_dir / "2025-08-30_walking_log.txt",
        output_dir / "2025-08-30_walking_log.csv",
        output_dir / "2025-08-30_walking_log.json",
        output_dir / "2025-08-30_walking_log.srt",
        output_dir / "2025-08-30_walking_log.vtt",
    ]
    expected_output_files_31 = [
        output_dir / "2025-08-31_walking_log.txt",
        output_dir / "2025-08-31_walking_log.csv",
        output_dir / "2025-08-31_walking_log.json",
        output_dir / "2025-08-31_walking_log.srt",
        output_dir / "2025-08-31_walking_log.vtt",
    ]

    for f in expected_output_files_30:
        assert f.exists(), f"Expected output file {f} for 2025-08-30 not found."
    for f in expected_output_files_31:
        assert f.exists(), f"Expected output file {f} for 2025-08-31 not found."

    # 生成されたCSVファイルの内容を軽く検証 (例: ヘッダー行が存在するか)
    csv_file_30 = output_dir / "2025-08-30_walking_log.csv"
    with open(csv_file_30, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) >= 3  # イベント名、空行、列名、データ行
        assert "イベント名" in lines[0]
        assert "start,end,speaker,text" in lines[2]

    csv_file_31 = output_dir / "2025-08-31_walking_log.csv"
    with open(csv_file_31, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) >= 3
        assert "イベント名" in lines[0]
        assert "start,end,speaker,text" in lines[2]

    # テキストファイルの内容を軽く検証 (例: SPEAKER_00が含まれるか)
    txt_file_30 = output_dir / "2025-08-30_walking_log.txt"
    with open(txt_file_30, "r", encoding="utf-8") as f:
        content = f.read()
        # assert "SPEAKER_00" in content or "SPEAKER_01" in content # 話者分離が行われるため

    txt_file_31 = output_dir / "2025-08-31_walking_log.txt"
    with open(txt_file_31, "r", encoding="utf-8") as f:
        content = f.read()
        # assert "SPEAKER_00" in content or "SPEAKER_01" in content # 話者分離が行われるため

    logging.info("Walking preset E2E test completed successfully.")


def test_group_files_by_date():
    """
    group_files_by_date関数が正しく日付でファイルをグループ化できるかテスト
    """
    files = [
        Path("walk_2025-08-30_part1.wav"),
        Path("walk_2025-08-30_part2.wav"),
        Path("rec_20250831.m4a"),
        Path("no_date_file.mp3"),
    ]
    grouped = group_files_by_date(files)
    assert "2025-08-30" in grouped
    assert len(grouped["2025-08-30"]) == 2
    assert "2025-08-31" in grouped
    assert len(grouped["2025-08-31"]) == 1
    assert "unknown" not in grouped  # 日付なしファイルは無視される
