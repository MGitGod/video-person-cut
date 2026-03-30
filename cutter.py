"""
cutter.py
検出した区間リストをもとに動画をカット・結合するモジュール。
ffmpeg が必要（sudo apt install ffmpeg）

【ブロックノイズ対策】
-c copy（ストリームコピー）は GOP 途中でカットするとブロックノイズが
発生する。セグメント切り出し時に libx264 で再エンコードすることで、
各セグメントの先頭を必ず Iフレーム（キーフレーム）にする。
結合時はすべてのセグメントが Iフレーム始まりなので -c copy で問題ない。
"""

import subprocess
import tempfile
from pathlib import Path


def cut_and_merge(
    video_path: str,
    intervals: list[tuple[float, float]],
    output_path: str,
    crf: int = 18,
    preset: str = "fast",
) -> None:
    """
    動画から指定した時間区間を切り出して1本に結合して保存する。

    Args:
        video_path:  入力動画のパス
        intervals:   [(開始秒, 終了秒), ...] のリスト
        output_path: 出力動画のパス
        crf:         映像品質（0〜51。低いほど高品質・大容量。デフォルト: 18）
        preset:      エンコード速度プリセット（デフォルト: fast）
                     ultrafast / superfast / veryfast / faster / fast /
                     medium / slow / slower / veryslow
    """
    if not intervals:
        print("切り出す区間がありませんでした。人物が検出されなかった可能性があります。")
        return

    # ffmpeg がインストールされているか確認
    _check_ffmpeg()

    # 一時ディレクトリに区間ごとの動画を保存
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        segment_paths: list[Path] = []

        print("\n動画を区間ごとに切り出しています...")
        print(f"エンコード設定: CRF={crf}, preset={preset}\n")

        for i, (start, end) in enumerate(intervals):
            segment_path = tmp / f"segment_{i:04d}.mp4"
            duration = end - start

            cmd = [
                "ffmpeg",
                "-y",                            # 上書き確認なし
                "-ss", str(start),               # 入力シーク（高速）
                "-i", video_path,                # 入力ファイル
                "-t", str(duration),             # 切り出し長さ（秒）
                # ── 映像: H.264 再エンコード ──────────────────────────
                # -c copy だとキーフレーム以外でカットした際に
                # ブロックノイズが出るため、必ず再エンコードする
                "-c:v", "libx264",
                "-crf", str(crf),                # 映像品質
                "-preset", preset,               # エンコード速度
                # セグメント先頭を必ず Iフレームにする
                "-force_key_frames", "expr:eq(n,0)",
                # ── 音声: AAC コピー or 変換 ─────────────────────────
                "-c:a", "aac",
                "-b:a", "192k",
                # タイムスタンプのずれを補正（結合時の音ズレ防止）
                "-avoid_negative_ts", "make_zero",
                str(segment_path),
            ]

            print(f"  区間 {i + 1}: {start:.2f}秒 〜 {end:.2f}秒 ({duration:.2f}秒)")
            _run(cmd)
            segment_paths.append(segment_path)

        # concat リストファイルを作成（ffmpeg の結合用）
        concat_list = tmp / "concat.txt"
        with concat_list.open("w") as f:
            for seg in segment_paths:
                # Windows パス対策でスラッシュに変換
                f.write(f"file '{seg.as_posix()}'\n")

        # セグメントを結合
        # 各セグメントが Iフレーム始まりなので -c copy で問題なく結合できる
        print("\nセグメントを結合しています...")
        merge_cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",            # 結合のみ、再エンコードなし（高速）
            output_path,
        ]
        _run(merge_cmd)

    print(f"\n完了！出力ファイル: {output_path}")


def _check_ffmpeg() -> None:
    """ffmpeg がインストールされているか確認する。"""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
        )
    except FileNotFoundError:
        raise EnvironmentError(
            "ffmpeg が見つかりません。\n"
            "以下のコマンドでインストールしてください:\n"
            "  sudo apt install ffmpeg"
        )


def _run(cmd: list[str]) -> None:
    """コマンドを実行してエラーを捕捉する。"""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg エラー:\n{result.stderr}"
        )
