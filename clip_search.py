"""
clip_search.py
CLIP を使って自然言語クエリで動画フレームを検索する実験スクリプト。

CLIP（Contrastive Language–Image Pre-Training）は OpenAI が開発した
画像とテキストを同じベクトル空間に埋め込むモデル。
テキストで「ステージで踊っている女性たち」のように映像の内容を説明すると
意味的に近いフレームを見つけることができる。

注意:
  - CLIP は英語で学習されているため、クエリは英語で入力する
  - 初回実行時にモデルが自動ダウンロードされる（約 600MB）
  - GPU なし・CPU のみでも動作する

使い方:
  uv run clip_search.py input.mp4 "women singing and dancing on stage"
  uv run clip_search.py input.mp4 "close up face portrait" --top-k 5
  uv run clip_search.py input.mp4 "crowd cheering" --frame-skip 15 --save-frames
  uv run clip_search.py input.mp4 "empty stage" --threshold 0.25
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


# ──────────────────────────────────────────────────────────────────────────────
# モデルのロード
# ──────────────────────────────────────────────────────────────────────────────

# Hugging Face のモデル名（初回実行時に自動ダウンロード）
_MODEL_NAME = "openai/clip-vit-base-patch32"


def _load_model() -> tuple[CLIPModel, CLIPProcessor]:
    """
    CLIP モデルとプロセッサを読み込む。
    初回実行時はモデルが Hugging Face Hub から自動ダウンロードされる（約 600MB）。
    2 回目以降はキャッシュから高速に読み込まれる。

    Returns:
        (model, processor) のタプル
    """
    print(f"CLIP モデルを読み込み中: {_MODEL_NAME}")
    print("  ※ 初回実行時はモデルが自動ダウンロードされます（約 600MB）")
    t = time.perf_counter()

    model     = CLIPModel.from_pretrained(_MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(_MODEL_NAME)
    model.eval()   # 推論モード（Dropout などを無効化）

    elapsed = time.perf_counter() - t
    print(f"  読み込み完了: {elapsed:.1f}秒\n")
    return model, processor


# ──────────────────────────────────────────────────────────────────────────────
# テキスト埋め込み
# ──────────────────────────────────────────────────────────────────────────────

def _encode_text(
    model: CLIPModel,
    processor: CLIPProcessor,
    query: str,
) -> torch.Tensor:
    """
    テキストクエリを正規化済みの特徴ベクトルに変換する。

    Args:
        model:     CLIP モデル
        processor: CLIP プロセッサ（トークナイザーを内包）
        query:     英語の検索クエリ

    Returns:
        shape (1, D) の正規化済み特徴ベクトル
    """
    inputs = processor(text=[query], return_tensors="pt", padding=True)
    with torch.no_grad():
        # get_text_features() は transformers のバージョンによって
        # テンソルではなく BaseModelOutputWithPooling を返すことがある。
        # text_model → text_projection と明示的に呼ぶことでバージョン差異を回避する。
        text_outputs = model.text_model(**inputs)
        features     = model.text_projection(text_outputs.pooler_output)
        # L2 正規化 → コサイン類似度 = ドット積 で計算できるようになる
        features = features / features.norm(dim=-1, keepdim=True)
    return features


# ──────────────────────────────────────────────────────────────────────────────
# バッチ処理
# ──────────────────────────────────────────────────────────────────────────────

def _process_batch(
    model: CLIPModel,
    processor: CLIPProcessor,
    text_features: torch.Tensor,
    batch_frames: list[np.ndarray],
    batch_secs: list[float],
    results: list[tuple[float, float, np.ndarray]],
) -> None:
    """
    フレームのバッチをまとめて画像ベクトルに変換し、
    テキストベクトルとのコサイン類似度を計算して results に追加する。

    バッチ処理にすることで、1 フレームずつ処理するより効率的になる。

    Args:
        model:         CLIP モデル
        processor:     CLIP プロセッサ
        text_features: テキストクエリの特徴ベクトル（shape: 1 × D）
        batch_frames:  RGB 画像の numpy 配列リスト
        batch_secs:    各フレームの動画内時刻（秒）リスト
        results:       結果を追記するリスト（インプレース更新）
    """
    pil_images = [Image.fromarray(f) for f in batch_frames]
    inputs     = processor(images=pil_images, return_tensors="pt", padding=True)

    with torch.no_grad():
        # vision_model → visual_projection と明示的に呼ぶことでバージョン差異を回避する
        vision_outputs = model.vision_model(**inputs)
        image_features = model.visual_projection(vision_outputs.pooler_output)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # shape: (1, D) × (D, N) → (1, N) → (N,)
    # テキストと各フレームのコサイン類似度を一括計算
    similarities = (text_features @ image_features.T).squeeze(0)

    for sim, sec, frame in zip(similarities.tolist(), batch_secs, batch_frames):
        results.append((float(sim), sec, frame))


# ──────────────────────────────────────────────────────────────────────────────
# メイン検索処理
# ──────────────────────────────────────────────────────────────────────────────

def search_video(
    video_path: str,
    query: str,
    frame_skip: int = 30,
    top_k: int = 10,
    batch_size: int = 8,
    threshold: float = 0.0,
) -> list[tuple[float, float, np.ndarray]]:
    """
    動画をフレームスキャンして、クエリと意味的に近いフレームを返す。

    処理の流れ:
      1. CLIP モデルをロード
      2. テキストクエリをベクトルに変換
      3. 動画をフレームごとにスキャン（frame_skip おきに処理）
      4. batch_size フレームをまとめて画像ベクトルに変換
      5. テキストと画像のコサイン類似度を計算
      6. 類似度の高い順に top_k 件を返す

    Args:
        video_path:  入力動画のパス
        query:       英語の検索クエリ
        frame_skip:  何フレームおきにスキャンするか
        top_k:       返す結果の件数
        batch_size:  一度にまとめて処理するフレーム数
        threshold:   この類似度以上のフレームのみを対象にする（0.0 = 絞り込みなし）

    Returns:
        [(類似度スコア, 秒数, RGBフレーム), ...] 類似度の高い順
    """
    model, processor = _load_model()

    print(f"クエリ:             {query}")
    t = time.perf_counter()
    text_features = _encode_text(model, processor, query)
    print(f"テキスト埋め込み:   {(time.perf_counter() - t) * 1000:.0f}ms\n")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"動画を開けませんでした: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_sec    = total_frames / fps
    scan_fps     = fps / frame_skip   # 実質的なスキャンレート

    print(f"動画情報:           {total_frames}フレーム / {fps:.2f}fps / {total_sec:.1f}秒")
    print(f"スキャン間隔:       {frame_skip}フレームおき（約 {scan_fps:.1f}fps 相当）")
    print(f"バッチサイズ:       {batch_size}")
    print(f"閾値:               {threshold}")
    print("\nスキャン開始...\n")

    results: list[tuple[float, float, np.ndarray]] = []
    batch_frames: list[np.ndarray] = []
    batch_secs:   list[float]      = []
    frame_idx  = 0
    processed  = 0
    scan_start = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch_frames.append(rgb)
            batch_secs.append(frame_idx / fps)

            # バッチが溜まったら処理する
            if len(batch_frames) >= batch_size:
                _process_batch(
                    model, processor, text_features,
                    batch_frames, batch_secs, results,
                )
                processed  += len(batch_frames)
                batch_frames = []
                batch_secs   = []

            progress = frame_idx / total_frames * 100
            print(f"\r進捗: {progress:.1f}% ({frame_idx}/{total_frames})", end="", flush=True)

        frame_idx += 1

    # 末尾に残ったフレームを処理する
    if batch_frames:
        _process_batch(
            model, processor, text_features,
            batch_frames, batch_secs, results,
        )
        processed += len(batch_frames)

    cap.release()

    scan_elapsed = time.perf_counter() - scan_start
    ms_per_frame = scan_elapsed / processed * 1000 if processed > 0 else 0.0
    print("\n\nスキャン完了")
    print(f"  処理フレーム数:   {processed}枚")
    print(f"  スキャン時間:     {scan_elapsed:.1f}秒")
    print(f"  1フレームあたり:  {ms_per_frame:.1f}ms")

    # 閾値でフィルタしてから類似度の高い順にソート
    filtered = [r for r in results if r[0] >= threshold]
    filtered.sort(key=lambda x: -x[0])
    return filtered[:top_k]


# ──────────────────────────────────────────────────────────────────────────────
# 結果の表示・保存
# ──────────────────────────────────────────────────────────────────────────────

def _fmt_sec(sec: float) -> str:
    """秒数を mm:ss.ss 形式に変換する。"""
    m = int(sec) // 60
    s = sec % 60
    return f"{m:02d}:{s:05.2f}"


def print_results(results: list[tuple[float, float, np.ndarray]]) -> None:
    """
    検索結果をコンソールに表示する。

    スコアをアスキーアートのバーで可視化し、
    類似度の分布が一目で把握できるようにする。
    """
    print(f"\n{'=' * 55}")
    print(f"  上位 {len(results)} 件の結果")
    print(f"{'=' * 55}")
    print(f"  {'順位':>4}  {'時刻':>8}  {'スコア':>6}  {'類似度バー'}")
    print(f"  {'─' * 4}  {'─' * 8}  {'─' * 6}  {'─' * 30}")

    if not results:
        print("  ヒットなし（--threshold を下げてみてください）")
        return

    # スコアの最大値に合わせてバーの長さを正規化する
    max_score = results[0][0]
    for rank, (score, sec, _) in enumerate(results, 1):
        bar_len = int(score / max(max_score, 0.01) * 28)
        bar     = "█" * bar_len + "░" * (28 - bar_len)
        print(f"  #{rank:>3}  {_fmt_sec(sec):>8}  {score:.4f}  {bar}")

    print(f"{'=' * 55}")


def save_result_frames(
    results: list[tuple[float, float, np.ndarray]],
    output_dir: str,
) -> None:
    """
    上位フレームを JPEG 画像として保存する。
    ファイル名に順位・時刻・スコアを含めることで後から見返しやすくする。

    Args:
        results:    search_video() の戻り値
        output_dir: 保存先フォルダのパス（存在しない場合は作成）
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n上位フレームを保存中: {out}/")
    for rank, (score, sec, frame_rgb) in enumerate(results, 1):
        m   = int(sec) // 60
        s   = sec % 60
        # ファイル名例: rank01_00m15.33s_score0.3142.jpg
        fname = out / f"rank{rank:02d}_{m:02d}m{s:05.2f}s_score{score:.4f}.jpg"
        bgr   = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(fname), bgr)
        print(f"  保存: {fname.name}")

    print(f"\n{len(results)} 枚を {out}/ に保存しました。")


# ──────────────────────────────────────────────────────────────────────────────
# CLI エントリーポイント
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLIP を使って自然言語で動画フレームを検索する実験スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使い方の例:
  uv run clip_search.py input.mp4 "women singing and dancing on stage"
  uv run clip_search.py input.mp4 "close up face portrait" --top-k 5
  uv run clip_search.py input.mp4 "crowd cheering" --frame-skip 15 --save-frames
  uv run clip_search.py input.mp4 "empty stage" --threshold 0.25

クエリのコツ:
  - 英語で入力する（CLIP は英語学習済みモデル）
  - 短い単語より「情景を説明する文」の方が精度が高い
      × "dance"  →  ○ "people dancing on a stage"
  - スコアの絶対値より「上位との差」を見る
      （スコア 0.30 でも他が 0.20 以下なら強いヒット）
        """,
    )

    parser.add_argument("input",
                        help="入力動画のパス（例: input.mp4）")
    parser.add_argument("query",
                        help="検索クエリ（英語推奨。例: 'women dancing on stage'）")
    parser.add_argument("--top-k", type=int, default=10,
                        help="上位何件を表示するか（デフォルト: 10）")
    parser.add_argument("--frame-skip", type=int, default=30,
                        help="何フレームおきにスキャンするか（デフォルト: 30 ≒ 1秒に1枚）")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="一度にまとめて処理するフレーム数（デフォルト: 8）")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="この類似度以上のフレームのみを対象にする（デフォルト: 0.0 = 絞り込みなし）")
    parser.add_argument("--output-dir", default="clip_results",
                        help="フレーム保存先フォルダ（デフォルト: clip_results/）")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"エラー: 動画ファイルが見つかりません: {input_path}")
        return

    print("=" * 55)
    print("  CLIP 動画フレーム検索")
    print("=" * 55)
    print(f"動画:               {input_path}")
    print(f"クエリ:             {args.query}")
    print(f"上位件数:           {args.top_k}")
    print(f"フレームスキップ:   {args.frame_skip}")
    print(f"バッチサイズ:       {args.batch_size}")
    print(f"閾値:               {args.threshold}")
    print(f"保存先:             {args.output_dir}/")
    print("=" * 55 + "\n")

    total_start = time.perf_counter()

    results = search_video(
        video_path = str(input_path),
        query      = args.query,
        frame_skip = args.frame_skip,
        top_k      = args.top_k,
        batch_size = args.batch_size,
        threshold  = args.threshold,
    )

    print_results(results)

    save_result_frames(results, args.output_dir)

    total_elapsed = time.perf_counter() - total_start
    print(f"\n合計処理時間: {total_elapsed:.1f}秒")


if __name__ == "__main__":
    main()