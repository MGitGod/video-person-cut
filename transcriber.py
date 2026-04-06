"""
transcriber.py
動画の音声を Whisper で文字起こしし、指定キーワードの出現時刻を検出するモジュール。

Whisper は動画ファイルを直接受け取れるため、音声の事前抽出は不要。
ffmpeg が PATH に通っていれば自動で音声デコードが行われる。
"""

import whisper


# ──────────────────────────────────────────────────────────────────────────────
# キーワードマーカーのカラーパレット
# ──────────────────────────────────────────────────────────────────────────────

# タイムライン上のキーワードマーカーに順番に割り当てる色（最大8種）
KW_PALETTE = [
    "#FFD700",  # ゴールド
    "#FF6B6B",  # コーラルレッド
    "#51CF66",  # グリーン
    "#FF922B",  # オレンジ
    "#CC5DE8",  # パープル
    "#22B8CF",  # シアン
    "#FF8787",  # ライトピンク
    "#94D82D",  # ライムグリーン
]


def build_kw_colors(keywords: list[str]) -> dict[str, str]:
    """
    キーワードリストから keyword → 色 の辞書を作る。
    キーワードが 9 個以上ある場合はパレットを循環する。

    Args:
        keywords: キーワードのリスト

    Returns:
        {"キーワード": "#RRGGBB", ...} の辞書
    """
    return {kw: KW_PALETTE[i % len(KW_PALETTE)] for i, kw in enumerate(keywords)}


def detect_keywords(
    video_path: str,
    keywords: list[str],
    model_size: str = "small",
) -> list[tuple[str, float]]:
    """
    動画の音声を Whisper で文字起こしし、キーワードの出現時刻を返す。

    処理の流れ:
      1. Whisper モデルをロード（初回は自動ダウンロード）
      2. 動画ファイルを直接文字起こし（音声抽出は Whisper が内部で実行）
      3. 各セグメントのテキストにキーワードが含まれるか検索
      4. ヒットした (キーワード, 開始秒) をリストで返す

    Args:
        video_path:  入力動画のパス
        keywords:    検索するキーワードのリスト
        model_size:  Whisper モデルサイズ
                     "tiny" / "base" / "small" / "medium" / "large"

    Returns:
        [(キーワード, 時刻秒), ...] を時刻順に並べたリスト。
        同一セグメントに複数のキーワードがある場合は両方を記録する。
    """
    if not keywords:
        return []

    print(f"【文字起こし】Whisper ({model_size}) を読み込み中...")
    print("  ※ 初回実行時はモデルが自動ダウンロードされます")
    model = whisper.load_model(model_size)

    # キーワードを initial_prompt に含めることで固有名詞の認識精度を上げる。
    # Whisper はこのテキストを「直前の発話」として参照するため、
    # キーワードの正しい表記をモデルに事前に教える効果がある。
    initial_prompt = "なぎちゃん、".join(keywords) + "乃木坂46。" if keywords else None

    print(f"【文字起こし】音声を解析中: {video_path}")
    if initial_prompt:
        print(f"  ヒント文: {initial_prompt}")
    result = model.transcribe(
        video_path,
        language="ja",          # 日本語固定（自動検出より精度が高い）
        verbose=False,          # Whisper 内部のログを抑制
        initial_prompt=initial_prompt,  # 固有名詞の表記ゆれを抑制
    )

    hits: list[tuple[str, float]] = []

    for segment in result["segments"]:
        text  = segment["text"]
        start = float(segment["start"])

        for kw in keywords:
            if kw in text:
                hits.append((kw, start))
                m = int(start) // 60
                s = start % 60
                print(f"  [{m:02d}:{s:05.2f}] 「{kw}」を検出 → {text.strip()}")

    # 時刻順にソート（同じ時刻ならキーワード名順）
    hits.sort(key=lambda x: (x[1], x[0]))

    print(f"【文字起こし】完了: {len(hits)} 件のキーワードを検出\n")
    return hits
