import argparse
import logging
import sys
import os
from pathlib import Path

from ladit_pipe.pipeline import execute_pipeline

# ロギング設定
def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - [%(levelname)s] - %(message)s',
        stream=sys.stdout
    )

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="高性能話者分離文字起こしシステム")
    parser.add_argument("input", type=Path, help="入力ファイルまたはディレクトリ")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default="output",
        help="出力ディレクトリ (デフォルト: output)",
    )
    parser.add_argument("--whisper-model", default="medium", help="Whisperモデル名")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"), help="Hugging Face トークン")
    parser.add_argument("--device", default="cuda", help="デバイス (cuda/cpu)")
    parser.add_argument("--resume", action="store_true", help="中断された処理を再開")
    parser.add_argument(
        "--min-speakers", type=int, default=2, help="最小話者数 (デフォルト: 2)"
    )
    parser.add_argument(
        "--max-speakers", type=int, default=10, help="最大話者数 (デフォルト: 10)"
    )
    parser.add_argument(
        "--sensitive",
        action="store_true",
        help="高感度モード（話者変化をより細かく検出）",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="walking",
        choices=["meeting", "walking"],
        help="処理プリセット (meeting/walking)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="デバッグレベルの詳細なログを出力する"
    )

    args = parser.parse_args()
    
    # logging設定を呼び出す
    setup_logging(args.verbose)
    
    # 出力ディレクトリ作成
    args.output.mkdir(parents=True, exist_ok=True)

    try:
        execute_pipeline(args)

    except KeyboardInterrupt:
        logger.info("\n処理が中断されました。--resume オプションで再開できます。")
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
