"""
detector.py
参考画像の人物が映っているフレームを検出するモジュール。

検出の優先順位:
  1. InsightFace による顔認識（最も精度が高い）
  2. YOLOv8n による人物検出（顔が見切れ・隠れた場合のフォールバック）

TensorFlow 不要で CPU でも動作する。
"""

from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


def _build_app() -> FaceAnalysis:
    """
    insightface の FaceAnalysis を初期化して返す。
    初回実行時にモデルが自動ダウンロードされる（約500MB）。
    """
    app = FaceAnalysis(
        name="buffalo_sc",
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def _build_yolo():
    """
    YOLOv8n（最軽量モデル）を初期化して返す。
    顔が検出できない場合の人物フォールバック検出に使用する。
    初回実行時にモデルが自動ダウンロードされる（約6MB）。
    """
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")


def load_reference_embeddings(references_dir: str) -> list[np.ndarray]:
    """
    参考画像フォルダ内の画像から顔の特徴量（埋め込みベクトル）を抽出する。

    Args:
        references_dir: 参考画像が入ったフォルダのパス

    Returns:
        特徴量ベクトルのリスト
    """
    ref_dir = Path(references_dir)
    if not ref_dir.exists():
        raise FileNotFoundError(f"参考画像フォルダが見つかりません: {ref_dir}")

    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_files = [
        p for p in ref_dir.iterdir()
        if p.suffix.lower() in image_extensions
    ]

    if not image_files:
        raise ValueError(
            f"参考画像が見つかりません: {ref_dir}\n"
            "jpg / png / webp などの画像を入れてください。"
        )

    print("顔認識モデルを初期化中...（初回は自動ダウンロードがあります）")
    app = _build_app()

    print(f"参考画像: {len(image_files)}枚 を読み込み中...")
    embeddings: list[np.ndarray] = []

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ✗ {img_path.name} → スキップ（画像を読み込めませんでした）")
            continue

        faces = app.get(img)
        if not faces:
            print(f"  ✗ {img_path.name} → スキップ（顔を検出できませんでした）")
            continue

        # 複数の顔が写っていても最初の1つだけ使う
        embeddings.append(faces[0].normed_embedding)
        print(f"  ✓ {img_path.name}")

    if not embeddings:
        raise ValueError(
            "参考画像から顔を1つも検出できませんでした。\n"
            "顔がはっきり写っている画像を用意してください。"
        )

    print(f"参考画像から {len(embeddings)} 件の顔特徴量を取得しました。\n")
    return embeddings


def detect_person_intervals(
    video_path: str,
    reference_embeddings: list[np.ndarray],
    threshold: float = 0.4,
    frame_skip: int = 5,
    padding_sec: float = 0.5,
    min_gap_sec: float = 1.0,
    use_body_fallback: bool = True,
) -> list[tuple[float, float]]:
    """
    動画を解析して、参考画像の人物が映っている時間区間のリストを返す。

    Args:
        video_path:            入力動画のパス
        reference_embeddings:  参考画像の特徴量リスト
        threshold:             コサイン類似度のしきい値（デフォルト: 0.4）
        frame_skip:            何フレームおきに検出するか
        padding_sec:           区間の前後に追加するパディング秒数
        min_gap_sec:           この秒数以内の消失は無視してつなぐ（デフォルト: 1.0秒）
                               YOLOフォールバックの継続時間にも同じ値を使用する
        use_body_fallback:     顔未検出時にYOLOv8nで人物検出をフォールバックとして使うか

    Returns:
        [(開始秒, 終了秒), ...] のリスト
    """
    print("顔認識モデルを初期化中...")
    app = _build_app()

    yolo = None
    if use_body_fallback:
        print("人物検出モデル（YOLOv8n）を初期化中...（初回は自動ダウンロードがあります）")
        yolo = _build_yolo()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"動画を開けませんでした: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_sec = total_frames / fps

    print(f"動画情報: {total_frames}フレーム / {fps:.2f}fps / {total_sec:.1f}秒")
    print(f"オプション: min_gap={min_gap_sec}秒 / "
          f"body_fallback={'有効' if use_body_fallback else '無効'}")
    print("検出記号の凡例: [✓]=顔認識一致  [~]=YOLOフォールバック一致  [ ]=不一致")
    print("顔認識による検出を開始します...\n")

    # フレームごとに該当人物の有無を記録
    person_flags: dict[int, bool] = {}

    # 最後に顔が一致したフレームインデックス（-1 は未検出）
    last_face_match_frame: int = -1

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            # ── フェーズ1: InsightFace による顔認識 ─────────────────────
            face_matched = _is_target_person_in_frame(
                frame, app, reference_embeddings, threshold
            )

            if face_matched:
                # 顔認識で一致 → 確定、最終一致フレームを更新
                matched = True
                last_face_match_frame = frame_idx

            elif yolo is not None and last_face_match_frame >= 0:
                # ── フェーズ2: YOLOv8n フォールバック ────────────────────
                # 最後に顔が検出されてから min_gap_sec 以内であれば
                # 人物検出を代用する（横顔・手隠しなどに対応）
                sec_since_face = (frame_idx - last_face_match_frame) / fps
                if sec_since_face <= min_gap_sec:
                    matched = _has_person_in_frame(frame, yolo)
                else:
                    matched = False
            else:
                matched = False

            person_flags[frame_idx] = matched

            # 進捗表示（検出方法を記号で区別）
            progress = frame_idx / total_frames * 100
            if face_matched:
                mark = "✓"
            elif matched:
                mark = "~"
            else:
                mark = " "
            print(
                f"\r進捗: {progress:.1f}% ({frame_idx}/{total_frames}) [{mark}]",
                end="",
            )

        frame_idx += 1

    cap.release()
    print("\n検出完了！")

    # スキップしたフレームを前のフレームの結果で埋める
    all_flags: list[bool] = []
    last_flag = False
    for i in range(total_frames):
        if i in person_flags:
            last_flag = person_flags[i]
        all_flags.append(last_flag)

    # 連続する「人物あり」フレームを時間区間にまとめる
    intervals = _flags_to_intervals(
        all_flags, fps, padding_sec, total_sec, min_gap_sec
    )

    # 結果を表示
    print(f"\n該当人物が映っている区間: {len(intervals)}個")
    for i, (start, end) in enumerate(intervals):
        duration = end - start
        print(f"  区間{i + 1}: {start:.2f}秒 〜 {end:.2f}秒（{duration:.2f}秒）")

    return intervals


def _is_target_person_in_frame(
    frame: np.ndarray,
    app: FaceAnalysis,
    reference_embeddings: list[np.ndarray],
    threshold: float,
) -> bool:
    """
    フレーム内の顔が参考画像の人物と一致するかを判定する。
    参考画像が複数ある場合、どれか1つと一致すればTrueを返す。

    Args:
        frame:                 OpenCVのフレーム（BGR形式）
        app:                   insightface の FaceAnalysis インスタンス
        reference_embeddings:  参考画像の特徴量リスト
        threshold:             コサイン類似度のしきい値

    Returns:
        一致する顔が見つかれば True
    """
    faces = app.get(frame)
    if not faces:
        return False

    for face in faces:
        face_embedding = face.normed_embedding
        for ref_embedding in reference_embeddings:
            # normed_embedding はすでに正規化済みなのでドット積 = コサイン類似度
            similarity = float(np.dot(face_embedding, ref_embedding))
            if similarity >= threshold:
                return True

    return False


def _has_person_in_frame(frame: np.ndarray, yolo_model) -> bool:
    """
    YOLOv8 を使ってフレーム内に人物（クラス0）がいるか確認する。
    顔認識が失敗した場合のフォールバックとして使用する。

    Args:
        frame:       OpenCVのフレーム（BGR形式）
        yolo_model:  初期化済みの YOLO モデル

    Returns:
        人物が1人以上検出されれば True
    """
    # classes=[0] で人物クラスのみに絞る、verbose=False でログ抑制
    results = yolo_model(frame, classes=[0], verbose=False)
    if not results:
        return False
    return len(results[0].boxes) > 0


def _flags_to_intervals(
    flags: list[bool],
    fps: float,
    padding_sec: float,
    total_sec: float,
    min_gap_sec: float = 0.0,
) -> list[tuple[float, float]]:
    """
    True/Falseのフラグリストを時間区間のリストに変換する。

    処理の流れ:
      1. min_gap_sec 未満の False 区間を True で補完（短い消失を無視）
      2. 連続する True を区間にまとめる
      3. 各区間に padding_sec を追加
      4. 重なった区間をマージ
    """
    if not any(flags):
        return []

    # ── ステップ1: 短い消失（False区間）を埋める ─────────────────────
    filled = _fill_short_gaps(flags, fps, min_gap_sec)

    raw_intervals: list[tuple[float, float]] = []
    in_person = False
    start_frame = 0

    for i, flag in enumerate(filled):
        if flag and not in_person:
            in_person = True
            start_frame = i
        elif not flag and in_person:
            in_person = False
            raw_intervals.append((start_frame / fps, i / fps))

    # 末尾まで人物ありの場合
    if in_person:
        raw_intervals.append((start_frame / fps, total_sec))

    # ── ステップ2: パディングを追加してから近い区間をマージ ──────────
    padded: list[tuple[float, float]] = []
    for start, end in raw_intervals:
        s = max(0.0, start - padding_sec)
        e = min(total_sec, end + padding_sec)
        padded.append((s, e))

    merged: list[tuple[float, float]] = []
    for start, end in padded:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    return merged


def _fill_short_gaps(
    flags: list[bool],
    fps: float,
    min_gap_sec: float,
) -> list[bool]:
    """
    True → False → True の区間で、False が続く長さが min_gap_sec 未満なら
    True で埋めて「短い消失を無視」する。

    例（fps=30, min_gap_sec=1.0 → min_gap_frames=30）:
      [T,T,F,F,F,T,T]  ← Falseが3フレーム（0.1秒）→ 埋める
      [T,T,F,...,F,T,T] ← Falseが40フレーム（1.3秒）→ 埋めない

    Args:
        flags:        True/False のフラグリスト
        fps:          動画のフレームレート
        min_gap_sec:  この秒数未満の消失は無視する

    Returns:
        補完後のフラグリスト（元のリストは変更しない）
    """
    if min_gap_sec <= 0:
        return flags[:]

    min_gap_frames = int(min_gap_sec * fps)
    result = flags[:]

    i = 0
    while i < len(result):
        if not result[i]:
            # False 区間の開始位置を記録
            gap_start = i
            while i < len(result) and not result[i]:
                i += 1
            gap_end = i  # result[gap_end] は True か末尾

            gap_len = gap_end - gap_start

            # 両端が True に挟まれた、かつ閾値未満のギャップなら True で埋める
            if (
                gap_start > 0
                and gap_end < len(result)
                and gap_len < min_gap_frames
            ):
                for j in range(gap_start, gap_end):
                    result[j] = True
        else:
            i += 1

    return result
