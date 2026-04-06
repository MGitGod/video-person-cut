"""
main.py
参考画像の人物が映っているシーンだけを切り出すCLIツール

使い方:
  python main.py input.mp4                                      # 検出 → GUI 起動
  python main.py input.mp4 --no-gui                             # 検出 → そのまま出力
  python main.py input.mp4 --load-intervals iv.json             # 再検出なしで GUI 起動
  python main.py input.mp4 --crf 23 --preset medium
  python main.py input.mp4 --min-gap 2.0
  python main.py input.mp4 --keywords 乃木坂46 サヨナラの意味   # キーワード検出あり
"""

import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path


def _x11_init_threads_early() -> None:
    """
    Linux/WSL 限定: ONNX Runtime・OpenCV 等がスレッドを使ったあとに
    Tk が X11 に接続すると XInitThreads 未呼び出しで xcb が abort することがある。
    他のライブラリより前に（できる限り早く）呼ぶ。
    Windows では何もしない。
    """
    if sys.platform != "linux":
        return
    try:
        import ctypes
        import ctypes.util
        lib_name = ctypes.util.find_library("X11")
        if not lib_name:
            return
        lib = ctypes.CDLL(lib_name)
        lib.XInitThreads.restype = ctypes.c_int
        lib.XInitThreads()
    except OSError:
        pass


_x11_init_threads_early()

from cutter import cut_and_merge # noqa: E402


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
    parser.add_argument("input",
                        help="入力動画ファイルのパス（例: input.mp4）")
    parser.add_argument("-r", "--references", default="references",
                        help="参考画像フォルダのパス（デフォルト: references/）")
    parser.add_argument("-o", "--output", default=None,
                        help="出力ファイルのパス（省略時は input_cut.mp4）")

    # ── 検出オプション ────────────────────────────────────────────────────────
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="顔の類似度しきい値 0.0〜1.0（デフォルト: 0.6）")
    parser.add_argument("--frame-skip", type=int, default=5,
                        help="何フレームおきに検出するか（デフォルト: 5）")
    parser.add_argument("--padding", type=float, default=0.5,
                        help="区間の前後に追加する余白秒数（デフォルト: 0.5秒）")
    parser.add_argument("--min-gap", type=float, default=1.0,
                        help="この秒数以内の消失は無視してつなぐ（デフォルト: 1.0秒）")
    parser.add_argument("--no-body-fallback", action="store_true",
                        help="YOLOv8n による人物検出フォールバックを無効にする")

    # ── エンコードオプション ──────────────────────────────────────────────────
    parser.add_argument("--crf", type=int, default=18,
                        help="映像品質（0〜51、低いほど高品質。デフォルト: 18）")
    parser.add_argument("--preset", default="fast",
                        choices=["ultrafast", "superfast", "veryfast", "faster",
                                 "fast", "medium", "slow", "slower", "veryslow"],
                        help="エンコード速度プリセット（デフォルト: fast）")

    # ── GUI / ワークフロー制御 ────────────────────────────────────────────────
    parser.add_argument("--no-gui", action="store_true",
                        help="GUI を起動せず検出後にそのまま出力する")
    parser.add_argument("--load-intervals", default=None, metavar="JSON_PATH",
                        help="保存済み区間 JSON を読み込んで再検出をスキップする")

    # ── キーワード検出オプション ──────────────────────────────────────────────
    parser.add_argument("--keywords", nargs="*", default=[], metavar="WORD",
                        help="音声から検出するキーワード（複数指定可、例: 乃木坂46 新内眞衣）")
    parser.add_argument("--whisper-model", default="small",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper モデルサイズ（デフォルト: small）")

    args = parser.parse_args()

    # 入力ファイルの確認
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"エラー: 動画ファイルが見つかりません: {input_path}")
        return

    output_path       = args.output or str(
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
    if args.keywords:
        print(f"キーワード:       {' / '.join(args.keywords)}")
        print(f"Whisper モデル:   {args.whisper_model}")
    print("=" * 50 + "\n")

    total_start    = time.perf_counter()
    step1_elapsed  = 0.0
    step2_elapsed  = 0.0
    step25_elapsed = 0.0   # 文字起こし
    step3_elapsed  = 0.0

    # ── 区間の取得（JSON 読み込み or 検出） ──────────────────────────────────
    keyword_hits: list[tuple[str, float]] = []

    if args.load_intervals:
        # JSON 読み込みモード（再検出スキップ）
        json_path = Path(args.load_intervals)
        if not json_path.exists():
            print(f"エラー: 区間ファイルが見つかりません: {json_path}")
            return
        data      = json.loads(json_path.read_text(encoding="utf-8"))
        intervals = [tuple(iv) for iv in data.get("intervals", [])]
        # Linux サブプロセス経由で渡されたキーワードヒットがあれば読み込む
        keyword_hits = [tuple(h) for h in data.get("keyword_hits", [])]
        print(f"区間ファイルを読み込みました: {json_path}（{len(intervals)}件）\n")

        # JSON に keyword_hits が入っていないのに --keywords が指定されていれば
        # Whisper だけ単独で実行する（人物検出は再実行しない）
        if args.keywords and not keyword_hits:
            from transcriber import detect_keywords
            print("【文字起こし】--keywords が指定されているため Whisper を実行します")
            t = time.perf_counter()
            keyword_hits = detect_keywords(
                video_path=str(input_path),
                keywords=args.keywords,
                model_size=args.whisper_model,
            )
            step25_elapsed = time.perf_counter() - t

    else:
        # detector は ONNX / OpenCV が重いため、JSON モード時は import しない
        from detector import detect_person_intervals, load_reference_embeddings

        # ── Thread 1: 参考画像読み込み → 人物検出（順番に実行） ──────────────
        def _detection_pipeline() -> list[tuple[float, float]]:
            """参考画像の読み込みから人物検出までを1つのスレッドで実行する。"""
            nonlocal step1_elapsed, step2_elapsed

            print("【人物検出】参考画像を読み込み中...")
            t1 = time.perf_counter()
            embs = load_reference_embeddings(args.references)
            step1_elapsed = time.perf_counter() - t1
            print(f"【人物検出】参考画像 完了: {_format_elapsed(step1_elapsed)}\n")

            print("【人物検出】フレーム解析中...")
            t2 = time.perf_counter()
            ivs = detect_person_intervals(
                video_path=str(input_path),
                reference_embeddings=embs,
                threshold=args.threshold,
                frame_skip=args.frame_skip,
                padding_sec=args.padding,
                min_gap_sec=args.min_gap,
                use_body_fallback=use_body_fallback,
            )
            step2_elapsed = time.perf_counter() - t2
            print(f"【人物検出】フレーム解析 完了: {_format_elapsed(step2_elapsed)}\n")
            return ivs

        # ── Thread 2: 音声文字起こし → キーワード検出 ────────────────────────
        def _transcription_pipeline() -> list[tuple[str, float]]:
            """Whisper でキーワードを検出する。--keywords 未指定時は即 return。"""
            nonlocal step25_elapsed
            if not args.keywords:
                return []
            from transcriber import detect_keywords
            t = time.perf_counter()
            hits = detect_keywords(
                video_path=str(input_path),
                keywords=args.keywords,
                model_size=args.whisper_model,
            )
            step25_elapsed = time.perf_counter() - t
            print(f"【文字起こし】完了: {_format_elapsed(step25_elapsed)}\n")
            return hits

        # ── 2スレッドを同時に走らせ、両方終わるまで待つ ──────────────────────
        if args.keywords:
            print("【並列処理開始】人物検出 + 音声文字起こし を同時に実行します\n")
        parallel_start = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            future_detect     = pool.submit(_detection_pipeline)
            future_transcribe = pool.submit(_transcription_pipeline)

            # result() は例外が起きていればここで再 raise される
            intervals    = future_detect.result()
            keyword_hits = future_transcribe.result()

        parallel_elapsed = time.perf_counter() - parallel_start
        if args.keywords:
            print(f"【並列処理完了】合計: {_format_elapsed(parallel_elapsed)}\n")

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
        # GUI なし → そのまま出力
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

    elif sys.platform == "linux" and not args.load_intervals:
        # Linux/WSL 限定: 検出直後は ONNX のワーカースレッドが残るため
        # 区間と keyword_hits を一時 JSON に書き出してクリーンな子プロセスで GUI を起動する。
        import subprocess
        import tempfile
        print("GUI を起動します（Linux: クリーンプロセスで起動）\n")
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, encoding="utf-8"
            ) as tmp:
                json.dump({
                    "intervals":    [list(iv) for iv in intervals],
                    "keyword_hits": [list(h)  for h  in keyword_hits],
                }, tmp)
                tmp_path = tmp.name
            cmd = [
                sys.executable, str(Path(__file__).resolve()),
                str(input_path),
                "--load-intervals", tmp_path,
                "-o", output_path,
                "--crf", str(args.crf),
                "--preset", args.preset,
            ]
            # キーワードリストも子プロセスに渡す（色順序の再現のため）
            if args.keywords:
                cmd += ["--keywords"] + args.keywords
            rc = subprocess.run(cmd, check=False).returncode
            if rc != 0:
                sys.exit(rc)
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    else:
        # Windows（または --load-intervals 指定時）: 直接 GUI を起動
        print("GUI を起動します。編集後に「✂ カット & 出力」ボタンで出力してください。\n")
        from editor import launch_editor
        launch_editor(
            video_path=str(input_path),
            intervals=intervals,
            output_path=output_path,
            crf=args.crf,
            preset=args.preset,
            keyword_hits=keyword_hits,
            keywords=args.keywords,
        )

    # ── 処理時間サマリー ──────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - total_start
    print("=" * 50)
    print("  処理時間サマリー")
    print("=" * 50)
    if not args.load_intervals:
        print(f"  参考画像読み込み:     {_format_elapsed(step1_elapsed)}")
        print(f"  人物検出:             {_format_elapsed(step2_elapsed)}")
        if args.keywords:
            print(f"  音声文字起こし:       {_format_elapsed(step25_elapsed)}")
            print(f"  ※ 上記2つは並列実行  (直列比 約{step2_elapsed + step25_elapsed:.1f}秒 → 実測 {_format_elapsed(parallel_elapsed)})")
    if args.no_gui:
        print(f"  カット＆結合:         {_format_elapsed(step3_elapsed)}")
    print(f"  {'─' * 29}")
    print(f"  合計:                 {_format_elapsed(total_elapsed)}")
    print("=" * 50)


if __name__ == "__main__":
    main()