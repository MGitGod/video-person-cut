"""
editor.py
検出区間を手動で確認・編集するデスクトップ GUI（Windows ネイティブ Tkinter）。

タイムライン:
  - Premiere 風フィルムストリップ（1秒 = 160px、横スクロール可能）
  - GUI 起動後、バックグラウンドスレッドで 1fps 分のサムネイルを順次生成
    → 生成できたコマから順番にグレーのプレースホルダーと差し替わる
  - 検出区間をスカイブルーのバーで表示
  - 区間の両端ハンドルをドラッグしてリサイズ、中央ドラッグで移動
  - ハンドルにカーソルを置くと ←→ カーソルに変わる
  - ハンドルドラッグ中にシークヘッドへ近づくと自動スナップ（吸着）
  - 空き領域をダブルクリックで 2 秒の新規区間を追加
  - 区間を右クリックで削除
  - タイムライン/プレビュー上クリックでシーク

機能:
  - 動画プレビュー再生（▶/⏸/シーク）
  - 左右矢印キーで 1 フレーム単位のシーク
  - 区間の開始・終了時刻を数値入力で微調整
  - Ctrl+Z / アンドゥボタンで操作を取り消す（最大 50 件）
  - 区間を JSON ファイルに保存・読み込み（再検出スキップ用）
  - 確定後にカット & 結合を実行
"""

import copy
import json
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageTk

from cutter import cut_and_merge


# ──────────────────────────────────────────────────────────────────────────────
# 定数
# ──────────────────────────────────────────────────────────────────────────────

PREVIEW_W = 640     # プレビュー表示幅（ピクセル）
PREVIEW_H = 360     # プレビュー表示高さ（ピクセル）
PLAY_FPS  = 30      # プレビュー再生フレームレート

# ── タイムライン ──────────────────────────────────────────────────────────────
THUMB_W  = 160      # サムネイル 1 枚の幅（= 1 秒あたりのピクセル数）
THUMB_H  = 90       # サムネイル 1 枚の高さ
BAR_H    = 24       # 区間オーバーレイバーの高さ
RULER_H  = 20       # 時刻目盛りの高さ
TL_H     = THUMB_H + BAR_H + RULER_H   # タイムライン Canvas の全高
EDGE_PX  = 8        # 区間ハンドルをつかむ許容幅（ピクセル）
SNAP_PX  = 12       # この距離（ピクセル）以内でシークヘッドにスナップする

# ── アンドゥ ──────────────────────────────────────────────────────────────────
MAX_UNDO = 50       # アンドゥ履歴の最大件数

OVERLAY_COLOR = "#1EA0FF"   # 検出区間バーの色（スカイブルー）
OVERLAY_DIM   = "#0D6EBD"   # 区間ハンドルの色（濃いブルー）
HEAD_COLOR    = "#FF4444"   # シークヘッドの色（赤）


# ──────────────────────────────────────────────────────────────────────────────
# ユーティリティ
# ──────────────────────────────────────────────────────────────────────────────

def _fmt(sec: float) -> str:
    """秒数を mm:ss.s 形式に変換する。"""
    m = int(sec) // 60
    s = sec % 60
    return f"{m:02d}:{s:04.1f}"


# ──────────────────────────────────────────────────────────────────────────────
# サムネイルキャッシュ
# ──────────────────────────────────────────────────────────────────────────────

class ThumbnailCache:
    """
    バックグラウンドスレッドで 1fps 分のサムネイルを順次生成してキャッシュする。

    Tkinter の制約:
      ImageTk.PhotoImage はメインスレッドでしか作成できない。
      そのため:
        - バックグラウンドスレッド → PIL Image として _pil に保存
        - メインスレッド（get_tk 呼び出し時）→ PhotoImage に変換して _tk に保存
    """

    def __init__(self, video_path: str, on_ready: Callable[[int], None]) -> None:
        """
        Args:
            video_path: 入力動画のパス
            on_ready:   サムネイル 1 枚が完成するたびに呼ぶコールバック（秒インデックスを渡す）。
                        バックグラウンドから呼ばれるので、受け取り側は
                        root.after(0, ...) でメインスレッドに戻すこと。
        """
        self._path     = video_path
        self._on_ready = on_ready
        self._pil: dict[int, Image.Image]        = {}   # バックグラウンドが書く
        self._tk:  dict[int, ImageTk.PhotoImage] = {}   # メインスレッドが書く
        self._lock   = threading.Lock()
        self._stop   = False
        self._thread = threading.Thread(target=self._generate, daemon=True)

    def start(self) -> None:
        """バックグラウンドでサムネイル生成を開始する。"""
        self._thread.start()

    def stop(self) -> None:
        """バックグラウンド生成を停止する（アプリ終了時に呼ぶ）。"""
        self._stop = True

    def get_tk(self, sec: int) -> "ImageTk.PhotoImage | None":
        """
        指定秒のサムネイル（PhotoImage）を返す。
        まだ生成されていなければ None を返す。
        必ずメインスレッドから呼ぶこと。
        """
        # すでに PhotoImage に変換済みならそのまま返す
        if sec in self._tk:
            return self._tk[sec]
        # PIL Image が届いていれば PhotoImage に変換する（メインスレッド限定）
        with self._lock:
            pil = self._pil.get(sec)
        if pil is None:
            return None
        tk_img = ImageTk.PhotoImage(pil)
        self._tk[sec] = tk_img
        return tk_img

    def _generate(self) -> None:
        """バックグラウンドスレッドの処理: 1fps 分を先頭から順番に生成する。"""
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            return
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_sec    = int(total_frames / fps)

        for sec in range(total_sec + 1):
            if self._stop:
                break
            # 各秒の先頭フレームを取得
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb).resize((THUMB_W, THUMB_H), Image.LANCZOS)

            with self._lock:
                self._pil[sec] = pil

            # コールバックはバックグラウンドから呼ぶ
            # → 受け取り側が root.after(0, ...) でメインスレッドに戻す
            self._on_ready(sec)

        cap.release()


# ──────────────────────────────────────────────────────────────────────────────
# データモデル
# ──────────────────────────────────────────────────────────────────────────────

class IntervalModel:
    """
    編集中の区間リストを管理するモデルクラス。
    変更があると登録されたコールバックをすべて呼ぶ（Observer パターン）。
    アンドゥ用の履歴スタックも内包する（スタック式、最大 MAX_UNDO 件）。
    """

    def __init__(
        self, intervals: list[tuple[float, float]], total_sec: float
    ) -> None:
        self._ivs: list[list[float]] = [list(iv) for iv in sorted(intervals)]
        self.total_sec = total_sec
        self._cbs: list[Callable] = []
        # アンドゥ用の履歴スタック
        # 各要素は _ivs のディープコピー（編集操作の直前の状態を積む）
        self._history: list[list[list[float]]] = []

    def add_listener(self, cb: Callable) -> None:
        self._cbs.append(cb)

    def _notify(self) -> None:
        for cb in self._cbs:
            cb()

    def _push_history(self) -> None:
        """
        現在の _ivs をアンドゥスタックに積む。
        編集操作の直前に呼ぶことで、その操作を取り消せるようにする。
        MAX_UNDO 件を超えた場合は最も古い履歴を捨てる。
        """
        self._history.append(copy.deepcopy(self._ivs))
        if len(self._history) > MAX_UNDO:
            self._history.pop(0)

    def can_undo(self) -> bool:
        """アンドゥできる操作が残っているか返す。"""
        return len(self._history) > 0

    def undo(self) -> bool:
        """
        1つ前の状態に戻す。
        履歴スタックから最新の状態を取り出して _ivs に戻し、リスナーに通知する。

        Returns:
            アンドゥできた場合 True、履歴がなく何もしなかった場合 False
        """
        if not self._history:
            return False
        self._ivs = self._history.pop()
        self._notify()
        return True

    def get(self) -> list[tuple[float, float]]:
        """現在の区間リストをタプルのリストで返す。"""
        return [tuple(iv) for iv in self._ivs]

    def __len__(self) -> int:
        return len(self._ivs)

    # ── 編集操作（すべて _push_history() → 変更 → _notify() の流れ） ─────────

    def set_start(self, idx: int, v: float) -> None:
        self._push_history()
        end = self._ivs[idx][1]
        self._ivs[idx][0] = max(0.0, min(v, end - 0.1))
        self._notify()

    def set_end(self, idx: int, v: float) -> None:
        self._push_history()
        start = self._ivs[idx][0]
        self._ivs[idx][1] = min(self.total_sec, max(v, start + 0.1))
        self._notify()

    def move(self, idx: int, delta: float) -> None:
        self._push_history()
        s, e = self._ivs[idx]
        dur  = e - s
        ns   = max(0.0, min(s + delta, self.total_sec - dur))
        self._ivs[idx] = [ns, ns + dur]
        self._notify()

    def add(self, start: float, end: float) -> None:
        self._push_history()
        self._ivs.append([max(0.0, start), min(self.total_sec, end)])
        self._ivs.sort()
        self._notify()

    def remove(self, idx: int) -> None:
        self._push_history()
        self._ivs.pop(idx)
        self._notify()

    def replace_all(self, intervals: list[tuple[float, float]]) -> None:
        """区間リストを丸ごと差し替える（JSON 読み込み時に使用）。"""
        self._push_history()
        self._ivs = [list(iv) for iv in intervals]
        self._notify()


# ──────────────────────────────────────────────────────────────────────────────
# フィルムストリップタイムライン
# ──────────────────────────────────────────────────────────────────────────────

class FilmstripTimeline(tk.Canvas):
    """
    Premiere 風フィルムストリップタイムライン。

    Canvas の y 座標レイアウト:
      [0, THUMB_H)               → サムネイル帯（1 枚 = 1 秒 = THUMB_W px）
      [THUMB_H, THUMB_H+BAR_H)   → 区間オーバーレイバー
      [THUMB_H+BAR_H, TL_H)      → 時刻目盛り

    タグ体系:
      "thumb_{sec}"  → 各サムネイル / プレースホルダー
      "overlay"      → 区間バー背景 + 色付き区間矩形
      "handle"       → 区間の左右ハンドル
      "ruler"        → 時刻目盛り
      "head"         → シークヘッド（赤い縦線）

    操作:
      左クリック（空き）      → シーク
      左クリック（区間中央）  → ドラッグで移動
      左クリック（区間端）    → ドラッグでリサイズ（シークヘッドにスナップ）
      ハンドルにカーソルを置く→ ←→ カーソルに変わる
      空き領域ダブルクリック  → 2 秒の新規区間を追加
      右クリック（区間上）    → 削除確認
    """

    def __init__(
        self,
        parent: tk.Widget,
        model: IntervalModel,
        cache: ThumbnailCache,
        total_sec: float,
        on_seek: Callable[[float], None],
        **kw,
    ) -> None:
        canvas_w = max(1, int(total_sec * THUMB_W))
        super().__init__(
            parent,
            height=TL_H,
            scrollregion=(0, 0, canvas_w, TL_H),
            bg="#1A1A1A",
            **kw,
        )
        self._model     = model
        self._cache     = cache
        self._total_sec = total_sec
        self._on_seek   = on_seek
        self._head_sec  = 0.0
        self._drag: dict | None = None

        # 全秒のプレースホルダーを描いてからオーバーレイを重ねる
        self._draw_placeholders()
        self._redraw_overlay()

        self.bind("<ButtonPress-1>",   self._press)
        self.bind("<B1-Motion>",       self._drag_move)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Double-Button-1>", self._dbl_click)
        self.bind("<Button-3>",        self._right_click)
        # マウス移動時にカーソル形状を更新する
        self.bind("<Motion>",          self._on_motion)

        # モデルが変わるたびにオーバーレイを描き直す
        model.add_listener(self._redraw_overlay)

    # ── 座標変換 ──────────────────────────────────────────────────────────────

    def _sec2x(self, sec: float) -> float:
        """秒 → Canvas x 座標（ピクセル）。"""
        return sec * THUMB_W

    def _x2sec(self, canvas_x: float) -> float:
        """Canvas x 座標 → 秒（0 〜 total_sec にクリップ）。"""
        return max(0.0, min(canvas_x / THUMB_W, self._total_sec))

    # ── 描画 ─────────────────────────────────────────────────────────────────

    def _draw_placeholders(self) -> None:
        """
        初期表示: 全秒分のプレースホルダー（暗いグレー矩形）を描く。
        バックグラウンドでサムネイルが生成されるまでの仮表示。
        """
        for sec in range(int(self._total_sec) + 1):
            x = self._sec2x(sec)
            self.create_rectangle(
                x, 0, x + THUMB_W, THUMB_H,
                fill="#2A2A2A", outline="#111",
                tags=f"thumb_{sec}",
            )

    def update_thumbnail(self, sec: int) -> None:
        """
        指定秒のプレースホルダーを実際のサムネイル画像に置き換える。
        ThumbnailCache からの通知を受けてメインスレッドから呼ぶ。
        """
        img = self._cache.get_tk(sec)
        if img is None:
            return
        tag = f"thumb_{sec}"
        self.delete(tag)
        self.create_image(
            self._sec2x(sec), 0,
            image=img, anchor="nw",
            tags=tag,
        )
        # サムネイルは z-order の最背面に置く（オーバーレイより後ろ）
        self.tag_lower(tag)

    def _redraw_overlay(self) -> None:
        """
        区間バー・ハンドル・時刻目盛り・シークヘッドを描き直す。
        IntervalModel が変わるたびに自動で呼ばれる。
        """
        self.delete("overlay")
        self.delete("handle")
        self.delete("ruler")
        self.delete("head")

        total_w = int(self._total_sec * THUMB_W)
        bar_y0  = THUMB_H
        bar_y1  = THUMB_H + BAR_H

        # ── 区間バーの背景（全体をダークグレーで塗る）────────────────────────
        self.create_rectangle(
            0, bar_y0, total_w, bar_y1,
            fill="#2A2A2A", outline="",
            tags="overlay",
        )

        # ── 検出区間バー + 両端ハンドル ──────────────────────────────────────
        for s, e in self._model.get():
            x1 = self._sec2x(s)
            x2 = self._sec2x(e)

            # 区間バー（スカイブルー、上下 2px の余白）
            self.create_rectangle(
                x1, bar_y0 + 2, x2, bar_y1 - 2,
                fill=OVERLAY_COLOR, outline="",
                tags="overlay",
            )

            # 左ハンドル（濃いブルーの縦バー）
            self.create_rectangle(
                x1 - EDGE_PX // 2, bar_y0,
                x1 + EDGE_PX // 2, bar_y1,
                fill=OVERLAY_DIM, outline="",
                tags="handle",
            )
            # 右ハンドル
            self.create_rectangle(
                x2 - EDGE_PX // 2, bar_y0,
                x2 + EDGE_PX // 2, bar_y1,
                fill=OVERLAY_DIM, outline="",
                tags="handle",
            )

        # ── 時刻目盛り ────────────────────────────────────────────────────────
        ruler_y = bar_y1
        # 目盛りは最大 20 本になるよう間隔を調整
        step = max(1, int(self._total_sec / 20))
        for t in range(0, int(self._total_sec) + 1, step):
            x = self._sec2x(t)
            self.create_line(x, ruler_y, x, TL_H, fill="#555", tags="ruler")
            self.create_text(
                x + 2, ruler_y + 2,
                text=_fmt(t), anchor="nw",
                fill="#888", font=("Helvetica", 7),
                tags="ruler",
            )

        # ── シークヘッド（最前面）────────────────────────────────────────────
        self._draw_head()

    def _draw_head(self) -> None:
        """シークヘッド（赤い縦線）だけを描く。最前面に置く。"""
        self.delete("head")
        hx = self._sec2x(self._head_sec)
        self.create_line(hx, 0, hx, TL_H, fill=HEAD_COLOR, width=2, tags="head")
        self.tag_raise("head")

    # ── ヒットテスト ──────────────────────────────────────────────────────────

    def _hit(self, cx: float) -> "tuple[int, str] | None":
        """
        Canvas x 座標がどの区間のどの部位に当たるかを返す。

        Returns:
            (区間インデックス, "left" | "right" | "body") または None
        """
        for i, (s, e) in enumerate(self._model.get()):
            x1 = self._sec2x(s)
            x2 = self._sec2x(e)
            if abs(cx - x1) <= EDGE_PX:
                return i, "left"
            if abs(cx - x2) <= EDGE_PX:
                return i, "right"
            if x1 < cx < x2:
                return i, "body"
        return None

    # ── カーソル変更 ──────────────────────────────────────────────────────────

    def _on_motion(self, ev: tk.Event) -> None:
        """
        マウス移動時にカーソル形状を更新する。
        バーゾーン内でハンドルの上にいるときだけ ←→ カーソルに変える。
        それ以外では標準カーソルに戻す。
        """
        cx = self.canvasx(ev.x)
        cy = self.canvasy(ev.y)

        if THUMB_H <= cy < THUMB_H + BAR_H:
            hit = self._hit(cx)
            if hit and hit[1] in ("left", "right"):
                # 左右リサイズカーソル（Windows / Linux 共通の名前）
                self.config(cursor="sb_h_double_arrow")
                return

        # ハンドル以外では標準カーソルに戻す
        self.config(cursor="")

    # ── マウスイベント ────────────────────────────────────────────────────────

    def _press(self, ev: tk.Event) -> None:
        cx = self.canvasx(ev.x)
        cy = self.canvasy(ev.y)

        # ── サムネイル帯 or 目盛りゾーン → 常にシーク ────────────────────────
        # 区間の x 範囲内であっても、バーゾーン以外では区間操作しない
        if not (THUMB_H <= cy < THUMB_H + BAR_H):
            self._on_seek(self._x2sec(cx))
            return

        # ── 区間バーゾーン ────────────────────────────────────────────────────
        hit = self._hit(cx)
        if hit:
            idx, part = hit
            self._drag = {"idx": idx, "part": part, "last_cx": cx}
            # ボディクリックは即シークもする（ハンドルは現在位置を保つ）
            if part == "body":
                self._on_seek(self._x2sec(cx))
        else:
            self._on_seek(self._x2sec(cx))

    def _drag_move(self, ev: tk.Event) -> None:
        """
        ドラッグ中の処理。
        ハンドル（left / right）のドラッグ時はシークヘッドへのスナップ判定も行う。
        スナップ: ドラッグ後の座標がシークヘッドから SNAP_PX ピクセル以内なら
                  強制的にシークヘッドの秒数に吸着する。
        """
        if not self._drag:
            return
        cx    = self.canvasx(ev.x)
        delta = (cx - self._drag["last_cx"]) / THUMB_W   # ピクセル差 → 秒差
        idx, part = self._drag["idx"], self._drag["part"]
        ivs = self._model.get()

        if part == "left":
            new_val = ivs[idx][0] + delta
            # シークヘッドへのスナップ判定
            new_val = self._snap_to_head(new_val)
            self._model.set_start(idx, new_val)

        elif part == "right":
            new_val = ivs[idx][1] + delta
            # シークヘッドへのスナップ判定
            new_val = self._snap_to_head(new_val)
            self._model.set_end(idx, new_val)

        else:
            # body ドラッグは区間全体を移動（スナップなし）
            self._model.move(idx, delta)

        self._drag["last_cx"] = cx

    def _snap_to_head(self, sec: float) -> float:
        """
        指定した秒数がシークヘッドから SNAP_PX ピクセル以内なら
        シークヘッドの秒数を返す。そうでなければ sec をそのまま返す。

        Args:
            sec: スナップ前の秒数

        Returns:
            スナップ後の秒数
        """
        head_x = self._sec2x(self._head_sec)
        new_x  = self._sec2x(sec)
        if abs(new_x - head_x) <= SNAP_PX:
            return self._head_sec
        return sec

    def _release(self, _: tk.Event) -> None:
        self._drag = None

    def _dbl_click(self, ev: tk.Event) -> None:
        cx = self.canvasx(ev.x)
        cy = self.canvasy(ev.y)
        # バーゾーン以外でのダブルクリックはシークのみ（区間追加しない）
        if not (THUMB_H <= cy < THUMB_H + BAR_H):
            return
        if self._hit(cx):
            return     # 既存区間上のダブルクリックは無視
        t = self._x2sec(cx)
        self._model.add(t, min(t + 2.0, self._total_sec))

    def _right_click(self, ev: tk.Event) -> None:
        cx = self.canvasx(ev.x)
        cy = self.canvasy(ev.y)
        # バーゾーン以外での右クリックは無視
        if not (THUMB_H <= cy < THUMB_H + BAR_H):
            return
        hit = self._hit(cx)
        if not hit:
            return
        idx, _ = hit
        if messagebox.askyesno("削除確認", f"区間 {idx + 1} を削除しますか？"):
            self._model.remove(idx)

    # ── 外部 API ──────────────────────────────────────────────────────────────

    def update_head(self, sec: float) -> None:
        """
        再生位置インジケーター（赤い縦線）を更新する。
        overlay 全体を再描画せず、head だけ差し替えるので軽い。
        """
        self._head_sec = sec
        self._draw_head()
        self._scroll_to_head(sec)

    def _scroll_to_head(self, sec: float) -> None:
        """
        シークヘッドが現在のビューポート外にあれば、
        ヘッドが中央に来るように横スクロールする。
        """
        if self._total_sec <= 0:
            return
        left_frac, right_frac = self.xview()
        viewport_w = right_frac - left_frac      # ビューポート幅（割合）
        hx_frac    = sec / self._total_sec       # ヘッド位置（割合）
        if hx_frac < left_frac or hx_frac > right_frac:
            new_left = max(0.0, hx_frac - viewport_w / 2)
            self.xview_moveto(new_left)


# ──────────────────────────────────────────────────────────────────────────────
# 区間リストウィジェット
# ──────────────────────────────────────────────────────────────────────────────

class IntervalListWidget(ttk.Frame):
    """
    区間の一覧表示 + 数値入力による微調整ウィジェット。
    モデルが変更されるたびに自動再描画される。
    """

    def __init__(
        self,
        parent: tk.Widget,
        model: IntervalModel,
        on_seek: Callable[[float], None],
        **kw,
    ) -> None:
        super().__init__(parent, **kw)
        self._model   = model
        self._on_seek = on_seek

        # ヘッダー
        hdr = ttk.Frame(self)
        hdr.pack(fill="x", pady=(0, 2))
        for text, w in [("#", 3), ("開始 (秒)", 10), ("終了 (秒)", 10), ("長さ (秒)", 10)]:
            ttk.Label(hdr, text=text, width=w).pack(side="left")

        # スクロール可能なリスト領域
        self._cv    = tk.Canvas(self, height=180, highlightthickness=0)
        sb          = ttk.Scrollbar(self, orient="vertical", command=self._cv.yview)
        self._inner = ttk.Frame(self._cv)
        self._cv.create_window((0, 0), window=self._inner, anchor="nw")
        self._cv.configure(yscrollcommand=sb.set)
        self._inner.bind(
            "<Configure>",
            lambda _: self._cv.configure(scrollregion=self._cv.bbox("all")),
        )
        self._cv.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # 追加ボタン
        bf = ttk.Frame(self)
        bf.pack(fill="x", pady=4)
        ttk.Button(bf, text="＋ 区間追加", command=self._add).pack(
            side="left", padx=2
        )

        model.add_listener(self._rebuild)
        self._rebuild()

    def _rebuild(self) -> None:
        """区間リストの行をすべて作り直す。"""
        for w in self._inner.winfo_children():
            w.destroy()

        for i, (s, e) in enumerate(self._model.get()):
            row = ttk.Frame(self._inner)
            row.pack(fill="x", pady=1)

            ttk.Label(row, text=str(i + 1), width=3).pack(side="left")

            # 開始時刻
            sv = tk.StringVar(value=f"{s:.2f}")
            se = ttk.Entry(row, textvariable=sv, width=10)
            se.pack(side="left", padx=4)
            se.bind("<Return>",   lambda _, idx=i, v=sv: self._apply_s(idx, v))
            se.bind("<FocusOut>", lambda _, idx=i, v=sv: self._apply_s(idx, v))

            # 終了時刻
            ev = tk.StringVar(value=f"{e:.2f}")
            ee = ttk.Entry(row, textvariable=ev, width=10)
            ee.pack(side="left", padx=4)
            ee.bind("<Return>",   lambda _, idx=i, v=ev: self._apply_e(idx, v))
            ee.bind("<FocusOut>", lambda _, idx=i, v=ev: self._apply_e(idx, v))

            # 長さ（読み取り専用）
            ttk.Label(row, text=f"{e - s:.2f}", width=10).pack(side="left")

            # 頭出し・削除ボタン
            ttk.Button(
                row, text="▶", width=2,
                command=lambda sec=s: self._on_seek(sec),
            ).pack(side="left")
            ttk.Button(
                row, text="✕", width=2,
                command=lambda idx=i: self._remove(idx),
            ).pack(side="left", padx=2)

    def _apply_s(self, idx: int, v: tk.StringVar) -> None:
        try:
            self._model.set_start(idx, float(v.get()))
        except ValueError:
            pass

    def _apply_e(self, idx: int, v: tk.StringVar) -> None:
        try:
            self._model.set_end(idx, float(v.get()))
        except ValueError:
            pass

    def _add(self) -> None:
        ivs   = self._model.get()
        start = (ivs[-1][1] + 1.0) if ivs else 0.0
        start = min(start, self._model.total_sec - 2.0)
        self._model.add(start, min(start + 2.0, self._model.total_sec))

    def _remove(self, idx: int) -> None:
        if messagebox.askyesno("削除確認", f"区間 {idx + 1} を削除しますか？"):
            self._model.remove(idx)


# ──────────────────────────────────────────────────────────────────────────────
# メインアプリケーション
# ──────────────────────────────────────────────────────────────────────────────

class EditorApp:
    """プレビュー・フィルムストリップタイムライン・区間リストを統合するメインアプリ。"""

    def __init__(
        self,
        root: tk.Tk,
        video_path: str,
        intervals: list[tuple[float, float]],
        output_path: str,
        crf: int = 18,
        preset: str = "fast",
    ) -> None:
        self.root        = root
        self.video_path  = video_path
        self.output_path = output_path
        self.crf         = crf
        self.preset      = preset

        # 動画情報を取得
        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"動画を開けませんでした: {video_path}")
        self._fps       = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames    = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._total_sec = total_frames / self._fps

        # 再生状態
        self._playing  = False
        self._cur_sec  = 0.0
        self._after_id: str | None = None

        self._model = IntervalModel(intervals, self._total_sec)

        # サムネイルキャッシュ
        # コールバックは root.after(0, ...) でメインスレッドに渡す
        self._thumb_cache = ThumbnailCache(
            video_path,
            on_ready=lambda sec: root.after(0, self._on_thumb_ready, sec),
        )

        root.title(f"区間エディタ — {Path(video_path).name}")
        root.resizable(True, True)
        self._build_ui()
        self._seek(0.0)

        # UI が完全に表示されてからサムネイル生成を開始する
        root.after(100, self._thumb_cache.start)

    def _on_thumb_ready(self, sec: int) -> None:
        """
        バックグラウンドスレッドからサムネイル生成完了の通知を受け、
        タイムライン上の対応コマを更新する。
        """
        self._tl.update_thumbnail(sec)

    # ── UI 構築 ───────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = self.root

        # ── 上部バー（ファイル情報・JSON 操作・アンドゥ）────────────────────
        top = ttk.Frame(root)
        top.pack(fill="x", padx=8, pady=4)
        ttk.Label(
            top,
            text=f"動画: {Path(self.video_path).name}　全長: {_fmt(self._total_sec)}",
        ).pack(side="left")
        ttk.Button(top, text="JSON 読込",
                   command=self._load_json).pack(side="right", padx=4)
        ttk.Button(top, text="JSON 保存",
                   command=self._save_json).pack(side="right", padx=4)
        # アンドゥボタン（Ctrl+Z のヒントを括弧内に表示）
        ttk.Button(top, text="↩ アンドゥ (Ctrl+Z)",
                   command=self._undo).pack(side="right", padx=4)

        # ── 中央（プレビュー + 区間リスト）──────────────────────────────────
        center = ttk.Frame(root)
        center.pack(fill="both", expand=True, padx=8)

        # プレビューキャンバス
        self._preview = tk.Canvas(center, width=PREVIEW_W, height=PREVIEW_H, bg="black")
        self._preview.pack(side="left", padx=(0, 8))

        # 区間リスト
        lf = ttk.LabelFrame(center, text="区間リスト")
        lf.pack(side="left", fill="both", expand=True)
        IntervalListWidget(lf, self._model, self._seek).pack(
            fill="both", expand=True, padx=4, pady=4
        )

        # ── フィルムストリップタイムライン ───────────────────────────────────
        tlf = ttk.LabelFrame(
            root,
            text="タイムライン  "
                 "（クリック: シーク　ドラッグ: 移動/リサイズ　"
                 "ダブルクリック: 区間追加　右クリック: 区間削除）",
        )
        tlf.pack(fill="x", padx=8, pady=4)

        self._tl = FilmstripTimeline(
            tlf,
            model=self._model,
            cache=self._thumb_cache,
            total_sec=self._total_sec,
            on_seek=self._seek,
        )
        self._tl.pack(fill="x", padx=4, pady=(4, 0))

        # 横スクロールバー
        tl_sb = ttk.Scrollbar(tlf, orient="horizontal", command=self._tl.xview)
        tl_sb.pack(fill="x", padx=4, pady=(0, 4))
        self._tl.configure(xscrollcommand=tl_sb.set)

        # ── 再生コントロール ──────────────────────────────────────────────────
        ctrl = ttk.Frame(root)
        ctrl.pack(fill="x", padx=8, pady=2)
        ttk.Button(ctrl, text="⏮ 先頭",
                   command=lambda: self._seek(0.0)).pack(side="left", padx=2)
        ttk.Button(ctrl, text="◀ -5秒",
                   command=lambda: self._seek(self._cur_sec - 5)).pack(
            side="left", padx=2
        )
        self._play_btn = ttk.Button(ctrl, text="▶ 再生",
                                    command=self._toggle_play)
        self._play_btn.pack(side="left", padx=2)
        ttk.Button(ctrl, text="▶ +5秒",
                   command=lambda: self._seek(self._cur_sec + 5)).pack(
            side="left", padx=2
        )
        self._pos_lbl = ttk.Label(ctrl, text=f"00:00.0 / {_fmt(self._total_sec)}")
        self._pos_lbl.pack(side="left", padx=12)

        # ← → キーについてのヒントラベル
        ttk.Label(ctrl, text="← → : 1フレーム移動", foreground="#888").pack(
            side="right", padx=8
        )

        # ── 出力バー ─────────────────────────────────────────────────────────
        out = ttk.Frame(root)
        out.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Label(out, text=f"出力先: {self.output_path}").pack(side="left")
        ttk.Button(out, text="✂ カット & 出力",
                   command=self._export).pack(side="right", padx=4)

        # ── キーバインド ──────────────────────────────────────────────────────
        # 左右矢印キー: 1 フレーム単位のシーク
        root.bind("<Left>",      self._key_seek_left)
        root.bind("<Right>",     self._key_seek_right)
        # Ctrl+Z: アンドゥ
        root.bind("<Control-z>", lambda _: self._undo())

    # ── キーボードシーク ──────────────────────────────────────────────────────

    def _is_entry_focused(self) -> bool:
        """
        現在フォーカスが Entry ウィジェットにあるか判定する。
        True の場合は矢印キーをシークに使わず、テキスト編集を優先する。
        """
        w = self.root.focus_get()
        return isinstance(w, (tk.Entry, ttk.Entry))

    def _key_seek_left(self, _: tk.Event) -> None:
        """左矢印キー: 1 フレーム分だけ前にシークする。"""
        # Entry 入力中はテキストカーソル移動を妨げないためシークしない
        if self._is_entry_focused():
            return
        self._seek(self._cur_sec - 1.0 / self._fps)

    def _key_seek_right(self, _: tk.Event) -> None:
        """右矢印キー: 1 フレーム分だけ後にシークする。"""
        if self._is_entry_focused():
            return
        self._seek(self._cur_sec + 1.0 / self._fps)

    # ── アンドゥ ──────────────────────────────────────────────────────────────

    def _undo(self) -> None:
        """
        1つ前の区間状態に戻す。
        履歴がない（これ以上戻れない）場合はビープ音で通知する。
        """
        if not self._model.undo():
            # これ以上戻れない場合にビープ音を鳴らす（Windows: bell）
            self.root.bell()

    # ── 再生制御 ──────────────────────────────────────────────────────────────

    def _seek(self, sec: float) -> None:
        """指定秒にシークして 1 フレームをプレビュー表示する。"""
        self._cur_sec = max(0.0, min(sec, self._total_sec))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(self._cur_sec * self._fps))
        ret, frame = self._cap.read()
        if ret:
            self._show(frame)
        self._tl.update_head(self._cur_sec)
        self._pos_lbl.config(
            text=f"{_fmt(self._cur_sec)} / {_fmt(self._total_sec)}"
        )

    def _show(self, frame: np.ndarray) -> None:
        """OpenCV フレームをプレビューキャンバスに描画する。"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale  = min(PREVIEW_W / w, PREVIEW_H / h)
        nw, nh = int(w * scale), int(h * scale)
        img = ImageTk.PhotoImage(Image.fromarray(cv2.resize(rgb, (nw, nh))))
        self._preview.delete("all")
        self._preview.create_image(
            PREVIEW_W // 2, PREVIEW_H // 2,
            image=img, anchor="center",
        )
        # GC で画像が消えないよう参照を保持する
        self._preview._img = img

    def _toggle_play(self) -> None:
        if self._playing:
            self._playing = False
            self._play_btn.config(text="▶ 再生")
            if self._after_id:
                self.root.after_cancel(self._after_id)
                self._after_id = None
        else:
            self._playing = True
            self._play_btn.config(text="⏸ 一時停止")
            self._play_loop()

    def _play_loop(self) -> None:
        """after() を使ったノンブロッキング再生ループ。"""
        if not self._playing:
            return
        ret, frame = self._cap.read()
        if not ret:
            self._playing = False
            self._play_btn.config(text="▶ 再生")
            return
        self._cur_sec = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        self._show(frame)
        self._tl.update_head(self._cur_sec)
        self._pos_lbl.config(
            text=f"{_fmt(self._cur_sec)} / {_fmt(self._total_sec)}"
        )
        self._after_id = self.root.after(int(1000 / PLAY_FPS), self._play_loop)

    # ── JSON 保存 / 読み込み ─────────────────────────────────────────────────

    def _save_json(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON ファイル", "*.json")],
            initialfile="intervals.json",
        )
        if not path:
            return
        data = {"video_path": self.video_path, "intervals": self._model.get()}
        Path(path).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        messagebox.showinfo("保存完了", f"区間を保存しました:\n{path}")

    def _load_json(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("JSON ファイル", "*.json")]
        )
        if not path:
            return
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        ivs  = [tuple(iv) for iv in data.get("intervals", [])]
        if not messagebox.askyesno(
            "読み込み確認",
            f"{len(ivs)} 件の区間を読み込みます。\n現在の区間は上書きされます。よろしいですか？",
        ):
            return
        self._model.replace_all(ivs)

    # ── カット & 出力 ─────────────────────────────────────────────────────────

    def _export(self) -> None:
        ivs = self._model.get()
        if not ivs:
            messagebox.showwarning("区間なし", "出力する区間がありません。")
            return
        if self._playing:
            self._toggle_play()

        path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 ファイル", "*.mp4")],
            initialfile=Path(self.output_path).name,
        )
        if not path:
            return

        prog = tk.Toplevel(self.root)
        prog.title("出力中...")
        prog.geometry("360x80")
        prog.resizable(False, False)
        prog.grab_set()
        ttk.Label(prog,
                  text="カット＆結合を実行中です。しばらくお待ちください...").pack(pady=24)
        prog.update()

        try:
            cut_and_merge(
                video_path=self.video_path,
                intervals=ivs,
                output_path=path,
                crf=self.crf,
                preset=self.preset,
            )
            prog.destroy()
            messagebox.showinfo("完了", f"出力完了:\n{path}")
        except Exception as exc:
            prog.destroy()
            messagebox.showerror("エラー", str(exc))

    def on_close(self) -> None:
        """ウィンドウを閉じるときにリソースを解放する。"""
        if self._playing:
            self._toggle_play()
        self._thumb_cache.stop()   # バックグラウンドスレッドを停止
        self._cap.release()
        self.root.destroy()


# ──────────────────────────────────────────────────────────────────────────────
# 外部エントリーポイント
# ──────────────────────────────────────────────────────────────────────────────

def launch_editor(
    video_path: str,
    intervals: list[tuple[float, float]],
    output_path: str,
    crf: int = 18,
    preset: str = "fast",
) -> None:
    """
    エディタ GUI を起動する。main.py から呼び出す。

    Args:
        video_path:  入力動画のパス
        intervals:   検出済み区間リスト
        output_path: デフォルト出力パス
        crf:         エンコード品質
        preset:      エンコード速度プリセット
    """
    root = tk.Tk()
    app  = EditorApp(root, video_path, intervals, output_path, crf, preset)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()