import argparse
import logging

# ロガー設定
logger = logging.getLogger(__name__)


def execute_pipeline(args: argparse.Namespace):
    """
    コマンドライン引数に基づいて適切な処理プリセットを呼び出します。

    Args:
        args (argparse.Namespace): コマンドラインからパースされた引数オブジェクト。
    """
    logger.info(f"パイプライン実行開始: プリセット = {args.preset}")

    if args.preset == "meeting":
        from ladit_pipe.presets import meeting

        meeting.run(args)
    elif args.preset == "walking":
        from ladit_pipe.presets import walking

        walking.run(args)
    else:
        logger.error(f"不明なプリセット: {args.preset}")
        raise ValueError(f"不明なプリセット: {args.preset}")

    logger.info("パイプライン実行完了")
