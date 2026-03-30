"""
main.py
参考画像の人物が映っているシーンだけを切り出すCLIツール

使い方:
  python main.py input.mp4                         # 検出 → GUI 起動
  python main.py input.mp4 --no-gui                # 検出 → そのまま出力
  python main.py input.mp4 --load-intervals iv.json  # 再検出なしで GUI 起動
  python main.py input.mp4 --crf 23 --preset medium
  python main.py input.mp4 --min-gap 2.0
"""

import argparse
import json
import time
from pathlib import Path

from detector import load_reference_embeddings, detect_person_intervals
from cutter import cut_and_merge


def _format_elapsed(seconds: float) -> str:
    """
    経過秒数を読みやすい文字列に変換する。
    60秒未満は「12.3秒」、以上は「1分23秒」形式で返す。
    """
    if seconds < 60:
        return f"{seconds:.1f}秒"
    m = int(seconds) // 60
    s = seconds % 60
    return f"{m}分{s:.1f}秒"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="参考画像の人物が映っているシーンだけを動画から切り出すツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── 基本引数 ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "input",
        help="入力動画ファイルのパス（例: input.mp4）",
    )
    parser.add_argument(
        "-r", "--references",
        default="references",
        help="参考画像フォルダのパス（デフォルト: references/）",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="出力ファイルのパス（省略時は input_cut.mp4）",
    )

    # ── 検出オプション ────────────────────────────────────────────────────────
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="顔の類似度しきい値 0.0〜1.0（デフォルト: 0.6）",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=5,
        help="何フレームおきに検出するか（デフォルト: 5）",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.5,
        help="区間の前後に追加する余白秒数（デフォルト: 0.5秒）",
    )
    parser.add_argument(
        "--min-gap",
        type=float,
        default=1.0,
        help="この秒数以内の消失は無視してつなぐ（デフォルト: 1.0秒）",
    )
    parser.add_argument(
        "--no-body-fallback",
        action="store_true",
        help="YOLOv8n による人物検出フォールバックを無効にする",
    )

    # ── エンコードオプション ──────────────────────────────────────────────────
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="映像品質（0〜51、低いほど高品質。デフォルト: 18）",
    )
    parser.add_argument(
        "--preset",
        default="fast",
        choices=[
            "ultrafast", "superfast", "veryfast", "faster",
            "fast", "medium", "slow", "slower", "veryslow",
        ],
        help="エンコード速度プリセット（デフォルト: fast）",
    )

    # ── GUI / ワークフロー制御 ────────────────────────────────────────────────
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="GUI を起動せず検出後にそのまま出力する",
    )
    parser.add_argument(
        "--load-intervals",
        default=None,
        metavar="JSON_PATH",
        help="保存済み区間 JSON を読み込んで再検出をスキップする",
    )

    args = parser.parse_args()

    # 入力ファイルの確認
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"エラー: 動画ファイルが見つかりません: {input_path}")
        return

    # 出力パスの決定（省略時は input_cut.mp4）
    output_path = args.output or str(
        input_path.parent / f"{input_path.stem}_cut{input_path.suffix}"
    )
    use_body_fallback = not args.no_body_fallback

    print("=" * 50)
    print("  特定人物シーン切り出しツール")
    print("=" * 50)
    print(f"入力動画:         {input_path}")
    print(f"参考画像フォルダ: {args.references}")
    print(f"出力先:           {output_path}")
    print(f"類似度しきい値:   {args.threshold}")
    print(f"フレームスキップ: {args.frame_skip}")
    print(f"パディング:       {args.padding}秒")
    print(f"ギャップ補完:     {args.min_gap}秒以内")
    print(f"人物検出補助:     {'有効（YOLOv8n）' if use_body_fallback else '無効'}")
    print(f"映像品質 (CRF):   {args.crf}")
    print(f"エンコード速度:   {args.preset}")
    print(f"GUI モード:       {'無効' if args.no_gui else '有効'}")
    print("=" * 50 + "\n")

    total_start    = time.perf_counter()
    step1_elapsed  = 0.0
    step2_elapsed  = 0.0
    step3_elapsed  = 0.0

    # ── 区間の取得（検出 or JSON 読み込み） ────────────────────────────────────
    if args.load_intervals:
        # ── JSON 読み込みモード（再検出スキップ） ─────────────────────────────
        json_path = Path(args.load_intervals)
        if not json_path.exists():
            print(f"エラー: 区間ファイルが見つかりません: {json_path}")
            return
        data      = json.loads(json_path.read_text(encoding="utf-8"))
        intervals = [tuple(iv) for iv in data.get("intervals", [])]
        print(f"区間ファイルを読み込みました: {json_path}（{len(intervals)}件）\n")

    else:
        # ── ステップ1: 参考画像の読み込み ─────────────────────────────────────
        print("【ステップ1】参考画像の読み込み")
        step1_start = time.perf_counter()

        reference_embeddings = load_reference_embeddings(args.references)

        step1_elapsed = time.perf_counter() - step1_start
        print(f"→ 完了: {_format_elapsed(step1_elapsed)}\n")

        # ── ステップ2: 人物検出 ────────────────────────────────────────────────
        print("【ステップ2】人物検出")
        step2_start = time.perf_counter()

        intervals = detect_person_intervals(
            video_path=str(input_path),
            reference_embeddings=reference_embeddings,
            threshold=args.threshold,
            frame_skip=args.frame_skip,
            padding_sec=args.padding,
            min_gap_sec=args.min_gap,
            use_body_fallback=use_body_fallback,
        )

        step2_elapsed = time.perf_counter() - step2_start
        print(f"→ 完了: {_format_elapsed(step2_elapsed)}\n")

        if not intervals:
            print(
                "該当する人物が検出されませんでした。\n"
                "以下を試してみてください:\n"
                "  --threshold を下げる（例: --threshold 0.4）\n"
                "  参考画像を追加・差し替える（顔が正面・鮮明なものが良い）"
            )
            return

    # ── GUI モード / CLI モードの分岐 ──────────────────────────────────────────
    if args.no_gui:
        # ── ステップ3: そのまま出力 ────────────────────────────────────────────
        print("【ステップ3】カット＆結合")
        step3_start = time.perf_counter()

        cut_and_merge(
            video_path=str(input_path),
            intervals=intervals,
            output_path=output_path,
            crf=args.crf,
            preset=args.preset,
        )

        step3_elapsed = time.perf_counter() - step3_start
        print(f"→ 完了: {_format_elapsed(step3_elapsed)}\n")

    else:
        # ── GUI 起動（編集 → 出力はGUI側で行う） ─────────────────────────────
        print("GUI を起動します。編集後に「✂ カット & 出力」ボタンで出力してください。\n")
        from editor import launch_editor
        launch_editor(
            video_path=str(input_path),
            intervals=intervals,
            output_path=output_path,
            crf=args.crf,
            preset=args.preset,
        )

    # ── 処理時間サマリー ──────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - total_start
    print("=" * 50)
    print("  処理時間サマリー")
    print("=" * 50)
    if not args.load_intervals:
        print(f"  ステップ1 参考画像読み込み: {_format_elapsed(step1_elapsed)}")
        print(f"  ステップ2 人物検出:         {_format_elapsed(step2_elapsed)}")
    if args.no_gui:
        print(f"  ステップ3 カット＆結合:     {_format_elapsed(step3_elapsed)}")
    print(f"  {'─' * 29}")
    print(f"  合計:                       {_format_elapsed(total_elapsed)}")
    print("=" * 50)


if __name__ == "__main__":
    main()