"""
editor.py
検出区間を手動で確認・編集するデスクトップGUI。

機能:
  - 動画プレビュー再生
  - タイムライン上でドラッグして区間を移動・リサイズ
  - 区間の追加・削除
  - 開始・終了時刻を数値入力で微調整
  - 区間を JSON ファイルに保存・読み込み
  - 確定後にカット&結合を実行
"""

import json
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

PREVIEW_W    = 640   # プレビュー表示幅（ピクセル）
PREVIEW_H    = 360   # プレビュー表示高さ（ピクセル）
TIMELINE_H   = 64    # タイムラインの高さ（ピクセル）
EDGE_GRAB_PX = 8     # 区間の端をドラッグするためのつかみ幅（ピクセル）
PLAYBACK_FPS = 30    # プレビュー再生のフレームレート


# ──────────────────────────────────────────────────────────────────────────────
# データモデル
# ──────────────────────────────────────────────────────────────────────────────

class IntervalModel:
    """
    編集中の区間リストを管理するモデルクラス。
    変更があると登録された on_change コールバックを呼ぶ。
    """

    def __init__(self, intervals: list[tuple[float, float]], total_sec: float):
        # 各区間を [start, end] のリストとして保持（時刻順にソート）
        self._intervals: list[list[float]] = [list(iv) for iv in sorted(intervals)]
        self.total_sec = total_sec
        self._callbacks: list[Callable] = []

    # ── コールバック登録 ──────────────────────────────────────────────────────

    def add_change_listener(self, callback: Callable) -> None:
        self._callbacks.append(callback)

    def _notify(self) -> None:
        for cb in self._callbacks:
            cb()

    # ── 読み取り ──────────────────────────────────────────────────────────────

    def get_intervals(self) -> list[tuple[float, float]]:
        return [tuple(iv) for iv in self._intervals]

    def __len__(self) -> int:
        return len(self._intervals)

    # ── 編集操作 ──────────────────────────────────────────────────────────────

    def set_start(self, idx: int, value: float) -> None:
        """区間 idx の開始時刻を設定する（最小幅 0.1 秒を保証）。"""
        end = self._intervals[idx][1]
        self._intervals[idx][0] = max(0.0, min(value, end - 0.1))
        self._notify()

    def set_end(self, idx: int, value: float) -> None:
        """区間 idx の終了時刻を設定する（最小幅 0.1 秒を保証）。"""
        start = self._intervals[idx][0]
        self._intervals[idx][1] = min(self.total_sec, max(value, start + 0.1))
        self._notify()

    def move(self, idx: int, delta_sec: float) -> None:
        """区間 idx を delta_sec 秒だけ平行移動する。"""
        start, end = self._intervals[idx]
        duration = end - start
        new_start = max(0.0, min(start + delta_sec, self.total_sec - duration))
        self._intervals[idx] = [new_start, new_start + duration]
        self._notify()

    def add(self, start: float, end: float) -> None:
        """新しい区間を追加して時刻順に並べ直す。"""
        self._intervals.append([max(0.0, start), min(self.total_sec, end)])
        self._intervals.sort()
        self._notify()

    def remove(self, idx: int) -> None:
        """区間 idx を削除する。"""
        self._intervals.pop(idx)
        self._notify()


# ──────────────────────────────────────────────────────────────────────────────
# タイムラインウィジェット
# ──────────────────────────────────────────────────────────────────────────────

class TimelineWidget(tk.Canvas):
    """
    タイムラインを描画し、区間のドラッグ編集をサポートする Canvas ウィジェット。

    操作:
      - 区間の中央をドラッグ       → 移動
      - 区間の左端をドラッグ       → 開始時刻を変更
      - 区間の右端をドラッグ       → 終了時刻を変更
      - 空き領域をダブルクリック   → 2秒の新規区間を追加
      - 区間を右クリック           → 削除確認ダイアログ
      - 空き領域をクリック         → その位置にシーク
    """

    # 区間の塗り色（複数区間を色で区別）
    COLORS = ["#4A90D9", "#E67E22", "#27AE60", "#9B59B6", "#E74C3C"]

    def __init__(self, parent, model: IntervalModel, on_seek: Callable, **kwargs):
        super().__init__(parent, height=TIMELINE_H, bg="#1E1E1E", **kwargs)
        self.model   = model
        self.on_seek = on_seek  # シーク時に呼ぶコールバック（秒を引数に取る）

        self._drag: dict | None = None   # ドラッグ中の状態管理
        self._current_sec: float = 0.0  # 再生位置インジケーターの現在値

        self.bind("<Configure>",       lambda _: self._redraw())
        self.bind("<ButtonPress-1>",   self._on_press)
        self.bind("<B1-Motion>",       self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Double-Button-1>", self._on_double_click)
        self.bind("<Button-3>",        self._on_right_click)

        model.add_change_listener(self._redraw)

    # ── 描画 ─────────────────────────────────────────────────────────────────

    def _redraw(self) -> None:
        """タイムライン全体を再描画する。"""
        self.delete("all")
        w = self.winfo_width()
        if w < 2 or self.model.total_sec <= 0:
            return

        h     = TIMELINE_H
        total = self.model.total_sec

        # 目盛り（全体を10分割）
        step = max(1, int(total / 10))
        for t in range(0, int(total) + 1, step):
            x = t / total * w
            self.create_line(x, h - 10, x, h, fill="#555555")
            self.create_text(x + 2, h - 12, text=_fmt_sec(t),
                             anchor="sw", fill="#888888", font=("Helvetica", 7))

        # 区間バー
        for i, (start, end) in enumerate(self.model.get_intervals()):
            color = self.COLORS[i % len(self.COLORS)]
            x1 = start / total * w
            x2 = end   / total * w
            self.create_rectangle(x1, 4, x2, h - 4,
                                  fill=color, outline="white", width=1,
                                  tags=(f"iv_{i}", "interval"))
            # 幅が十分あれば番号を表示
            if x2 - x1 > 20:
                self.create_text((x1 + x2) / 2, h / 2,
                                 text=str(i + 1), fill="white",
                                 font=("Helvetica", 9, "bold"),
                                 tags=(f"iv_{i}", "interval"))

        # 再生位置インジケーター（赤い縦線）
        px = self._current_sec / total * w
        self.create_line(px, 0, px, h, fill="#FF4444", width=2, tags="playhead")

    # ── 座標 → 時刻 変換 ────────────────────────────────────────────────────

    def _x_to_sec(self, x: float) -> float:
        w = self.winfo_width()
        return max(0.0, min(x / w * self.model.total_sec, self.model.total_sec))

    # ── ヒットテスト ──────────────────────────────────────────────────────────

    def _hit_test(self, x: float) -> tuple[int, str] | None:
        """
        x ピクセル位置がどの区間のどの部位に当たるかを返す。

        Returns:
            (区間インデックス, "left" | "body" | "right") または None
        """
        w     = self.winfo_width()
        total = self.model.total_sec
        for i, (start, end) in enumerate(self.model.get_intervals()):
            x1 = start / total * w
            x2 = end   / total * w
            if x1 - EDGE_GRAB_PX <= x <= x1 + EDGE_GRAB_PX:
                return (i, "left")
            if x2 - EDGE_GRAB_PX <= x <= x2 + EDGE_GRAB_PX:
                return (i, "right")
            if x1 < x < x2:
                return (i, "body")
        return None

    # ── マウスイベントハンドラ ────────────────────────────────────────────────

    def _on_press(self, event: tk.Event) -> None:
        hit = self._hit_test(event.x)
        if hit:
            self._drag = {"idx": hit[0], "part": hit[1], "last_x": event.x}
        else:
            # 区間外クリック → シーク
            self.on_seek(self._x_to_sec(event.x))
            self._drag = None

    def _on_drag(self, event: tk.Event) -> None:
        if self._drag is None:
            return
        dx        = event.x - self._drag["last_x"]
        delta_sec = dx / self.winfo_width() * self.model.total_sec
        idx       = self._drag["idx"]
        part      = self._drag["part"]
        intervals = self.model.get_intervals()

        if part == "left":
            self.model.set_start(idx, intervals[idx][0] + delta_sec)
        elif part == "right":
            self.model.set_end(idx, intervals[idx][1] + delta_sec)
        elif part == "body":
            self.model.move(idx, delta_sec)

        self._drag["last_x"] = event.x

    def _on_release(self, event: tk.Event) -> None:
        self._drag = None

    def _on_double_click(self, event: tk.Event) -> None:
        """空き領域のダブルクリック → 2秒の新規区間を追加する。"""
        if self._hit_test(event.x):
            return  # 既存区間の上はスキップ
        t = self._x_to_sec(event.x)
        self.model.add(t, min(t + 2.0, self.model.total_sec))

    def _on_right_click(self, event: tk.Event) -> None:
        """区間の右クリック → 確認ダイアログを出して削除する。"""
        hit = self._hit_test(event.x)
        if hit is None:
            return
        idx, _ = hit
        if messagebox.askyesno("削除確認", f"区間 {idx + 1} を削除しますか？"):
            self.model.remove(idx)

    # ── 外部から呼ぶ API ─────────────────────────────────────────────────────

    def update_playhead(self, current_sec: float) -> None:
        """再生位置インジケーターを更新して再描画する。"""
        self._current_sec = current_sec
        self._redraw()


# ──────────────────────────────────────────────────────────────────────────────
# 区間リストウィジェット
# ──────────────────────────────────────────────────────────────────────────────

class IntervalListWidget(ttk.Frame):
    """
    区間の一覧を表示し、数値入力で開始・終了時刻を微調整できるウィジェット。
    モデルが変更されるたびに自動的に再描画される。
    """

    def __init__(self, parent, model: IntervalModel, on_seek: Callable, **kwargs):
        super().__init__(parent, **kwargs)
        self.model   = model
        self.on_seek = on_seek

        # ヘッダー行
        header = ttk.Frame(self)
        header.pack(fill="x", pady=(0, 2))
        ttk.Label(header, text="#",       width=3).pack(side="left")
        ttk.Label(header, text="開始 (秒)", width=10).pack(side="left", padx=4)
        ttk.Label(header, text="終了 (秒)", width=10).pack(side="left", padx=4)
        ttk.Label(header, text="長さ (秒)", width=10).pack(side="left", padx=4)

        # スクロール可能なリスト領域
        self._canvas  = tk.Canvas(self, height=180, highlightthickness=0)
        scrollbar     = ttk.Scrollbar(self, orient="vertical",
                                      command=self._canvas.yview)
        self._inner   = ttk.Frame(self._canvas)

        self._canvas.create_window((0, 0), window=self._inner, anchor="nw")
        self._canvas.configure(yscrollcommand=scrollbar.set)
        self._inner.bind("<Configure>", lambda _: self._canvas.configure(
            scrollregion=self._canvas.bbox("all")
        ))
        self._canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 区間追加ボタン
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", pady=4)
        ttk.Button(btn_frame, text="＋ 区間追加",
                   command=self._add_interval).pack(side="left", padx=2)

        model.add_change_listener(self._rebuild)
        self._rebuild()

    def _rebuild(self) -> None:
        """区間リストを全行作り直す。"""
        for widget in self._inner.winfo_children():
            widget.destroy()

        for i, (start, end) in enumerate(self.model.get_intervals()):
            row = ttk.Frame(self._inner)
            row.pack(fill="x", pady=1)

            ttk.Label(row, text=str(i + 1), width=3).pack(side="left")

            # 開始時刻 入力欄
            start_var = tk.StringVar(value=f"{start:.2f}")
            start_ent = ttk.Entry(row, textvariable=start_var, width=10)
            start_ent.pack(side="left", padx=4)
            start_ent.bind("<Return>",
                           lambda e, idx=i, v=start_var: self._apply_start(idx, v))
            start_ent.bind("<FocusOut>",
                           lambda e, idx=i, v=start_var: self._apply_start(idx, v))

            # 終了時刻 入力欄
            end_var = tk.StringVar(value=f"{end:.2f}")
            end_ent = ttk.Entry(row, textvariable=end_var, width=10)
            end_ent.pack(side="left", padx=4)
            end_ent.bind("<Return>",
                         lambda e, idx=i, v=end_var: self._apply_end(idx, v))
            end_ent.bind("<FocusOut>",
                         lambda e, idx=i, v=end_var: self._apply_end(idx, v))

            # 長さ（読み取り専用ラベル）
            ttk.Label(row, text=f"{end - start:.2f}", width=10).pack(side="left")

            # 頭出し再生・削除ボタン
            ttk.Button(row, text="▶", width=2,
                       command=lambda s=start: self.on_seek(s)).pack(side="left")
            ttk.Button(row, text="✕", width=2,
                       command=lambda idx=i: self._remove(idx)).pack(side="left", padx=2)

    def _apply_start(self, idx: int, var: tk.StringVar) -> None:
        try:
            self.model.set_start(idx, float(var.get()))
        except ValueError:
            pass

    def _apply_end(self, idx: int, var: tk.StringVar) -> None:
        try:
            self.model.set_end(idx, float(var.get()))
        except ValueError:
            pass

    def _add_interval(self) -> None:
        """末尾区間の直後に 2 秒の新規区間を追加する。"""
        intervals = self.model.get_intervals()
        if intervals:
            start = min(intervals[-1][1] + 1.0, self.model.total_sec - 2.0)
        else:
            start = 0.0
        self.model.add(start, min(start + 2.0, self.model.total_sec))

    def _remove(self, idx: int) -> None:
        if messagebox.askyesno("削除確認", f"区間 {idx + 1} を削除しますか？"):
            self.model.remove(idx)


# ──────────────────────────────────────────────────────────────────────────────
# メインアプリケーション
# ──────────────────────────────────────────────────────────────────────────────

class EditorApp:
    """
    プレビュー・タイムライン・区間リストを統合するメインアプリクラス。
    """

    def __init__(
        self,
        root: tk.Tk,
        video_path: str,
        intervals: list[tuple[float, float]],
        output_path: str,
        crf: int = 18,
        preset: str = "fast",
    ):
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
        self._playing    = False
        self._current_sec = 0.0
        self._after_id: str | None = None

        # データモデル
        self.model = IntervalModel(intervals, self._total_sec)

        root.title(f"区間エディタ — {Path(video_path).name}")
        root.resizable(True, True)
        self._build_ui()
        self._seek(0.0)

    # ── UI 構築 ───────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = self.root

        # ── 上部バー ──────────────────────────────────────────────────────────
        top = ttk.Frame(root)
        top.pack(fill="x", padx=8, pady=4)
        ttk.Label(top,
                  text=f"動画: {Path(self.video_path).name}　"
                       f"全長: {_fmt_sec(self._total_sec)}　"
                       f"区間数: {len(self.model)}").pack(side="left")
        ttk.Button(top, text="JSON 読込",
                   command=self._load_json).pack(side="right", padx=4)
        ttk.Button(top, text="JSON 保存",
                   command=self._save_json).pack(side="right", padx=4)

        # ── 中央: プレビュー + 区間リスト ─────────────────────────────────────
        center = ttk.Frame(root)
        center.pack(fill="both", expand=True, padx=8)

        # プレビュー
        self._preview = tk.Canvas(center,
                                  width=PREVIEW_W, height=PREVIEW_H,
                                  bg="black")
        self._preview.pack(side="left", padx=(0, 8))

        # 区間リスト
        list_frame = ttk.LabelFrame(center, text="区間リスト")
        list_frame.pack(side="left", fill="both", expand=True)
        IntervalListWidget(list_frame, self.model,
                           self._seek).pack(fill="both", expand=True,
                                            padx=4, pady=4)

        # ── タイムライン ──────────────────────────────────────────────────────
        tl_frame = ttk.LabelFrame(
            root,
            text="タイムライン（ドラッグ: 移動/リサイズ　"
                 "ダブルクリック: 追加　右クリック: 削除）",
        )
        tl_frame.pack(fill="x", padx=8, pady=4)
        self._timeline = TimelineWidget(tl_frame, self.model, self._seek)
        self._timeline.pack(fill="x", padx=4, pady=4)

        # ── 再生コントロール ──────────────────────────────────────────────────
        ctrl = ttk.Frame(root)
        ctrl.pack(fill="x", padx=8, pady=2)

        ttk.Button(ctrl, text="⏮ 先頭",
                   command=lambda: self._seek(0.0)).pack(side="left", padx=2)
        ttk.Button(ctrl, text="◀ -5秒",
                   command=lambda: self._seek(
                       self._current_sec - 5)).pack(side="left", padx=2)

        self._play_btn = ttk.Button(ctrl, text="▶ 再生",
                                    command=self._toggle_play)
        self._play_btn.pack(side="left", padx=2)

        ttk.Button(ctrl, text="▶ +5秒",
                   command=lambda: self._seek(
                       self._current_sec + 5)).pack(side="left", padx=2)

        self._pos_label = ttk.Label(
            ctrl, text=f"00:00.0 / {_fmt_sec(self._total_sec)}"
        )
        self._pos_label.pack(side="left", padx=12)

        # ── 出力 ──────────────────────────────────────────────────────────────
        out = ttk.Frame(root)
        out.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Label(out, text=f"出力先: {self.output_path}").pack(side="left")
        ttk.Button(out, text="✂ カット & 出力",
                   command=self._export).pack(side="right", padx=4)

    # ── 再生制御 ──────────────────────────────────────────────────────────────

    def _seek(self, sec: float) -> None:
        """指定秒数にシークして1フレームを表示する。"""
        self._current_sec = max(0.0, min(sec, self._total_sec))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(self._current_sec * self._fps))
        ret, frame = self._cap.read()
        if ret:
            self._show_frame(frame)
        self._timeline.update_playhead(self._current_sec)
        self._pos_label.config(
            text=f"{_fmt_sec(self._current_sec)} / {_fmt_sec(self._total_sec)}"
        )

    def _show_frame(self, frame: np.ndarray) -> None:
        """OpenCV フレームをプレビューキャンバスに表示する。"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        # アスペクト比を保ってリサイズ
        scale   = min(PREVIEW_W / w, PREVIEW_H / h)
        new_w   = int(w * scale)
        new_h   = int(h * scale)
        resized = cv2.resize(rgb, (new_w, new_h))
        img     = ImageTk.PhotoImage(Image.fromarray(resized))

        self._preview.delete("all")
        self._preview.create_image(
            PREVIEW_W // 2, PREVIEW_H // 2,
            image=img, anchor="center",
        )
        # ImageTk.PhotoImage は参照がなくなると GC で消えるため保持する
        self._preview._img_ref = img

    def _toggle_play(self) -> None:
        """再生・一時停止を切り替える。"""
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
        """after() を使ったノンブロッキングな再生ループ。"""
        if not self._playing:
            return

        ret, frame = self._cap.read()
        if not ret:
            # 末尾に達したら停止
            self._playing = False
            self._play_btn.config(text="▶ 再生")
            return

        self._current_sec = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        self._show_frame(frame)
        self._timeline.update_playhead(self._current_sec)
        self._pos_label.config(
            text=f"{_fmt_sec(self._current_sec)} / {_fmt_sec(self._total_sec)}"
        )
        self._after_id = self.root.after(
            int(1000 / PLAYBACK_FPS), self._play_loop
        )

    # ── JSON 保存 / 読み込み ─────────────────────────────────────────────────

    def _save_json(self) -> None:
        """区間リストを JSON ファイルに保存する。"""
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON ファイル", "*.json")],
            initialfile="intervals.json",
        )
        if not path:
            return
        data = {
            "video_path": self.video_path,
            "intervals":  self.model.get_intervals(),
        }
        Path(path).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        messagebox.showinfo("保存完了", f"区間を保存しました:\n{path}")

    def _load_json(self) -> None:
        """JSON ファイルから区間リストを読み込んでモデルを上書きする。"""
        path = filedialog.askopenfilename(
            filetypes=[("JSON ファイル", "*.json")]
        )
        if not path:
            return
        data      = json.loads(Path(path).read_text(encoding="utf-8"))
        intervals = [tuple(iv) for iv in data.get("intervals", [])]
        if not messagebox.askyesno(
            "読み込み確認",
            f"{len(intervals)} 件の区間を読み込みます。\n"
            "現在の区間は上書きされます。よろしいですか？",
        ):
            return
        self.model._intervals = [list(iv) for iv in intervals]
        self.model._notify()

    # ── カット & 出力 ─────────────────────────────────────────────────────────

    def _export(self) -> None:
        """確定した区間でカット&結合を実行する。"""
        intervals = self.model.get_intervals()
        if not intervals:
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

        # 進捗ウィンドウ
        prog = tk.Toplevel(self.root)
        prog.title("出力中...")
        prog.geometry("320x80")
        prog.grab_set()
        ttk.Label(prog,
                  text="カット&結合を実行中です。しばらくお待ちください...").pack(pady=24)
        prog.update()

        try:
            cut_and_merge(
                video_path=self.video_path,
                intervals=intervals,
                output_path=path,
                crf=self.crf,
                preset=self.preset,
            )
            prog.destroy()
            messagebox.showinfo("完了", f"出力完了:\n{path}")
        except Exception as e:
            prog.destroy()
            messagebox.showerror("エラー", str(e))

    def on_close(self) -> None:
        """ウィンドウを閉じるときにリソースを解放する。"""
        if self._playing:
            self._toggle_play()
        self._cap.release()
        self.root.destroy()


# ──────────────────────────────────────────────────────────────────────────────
# ユーティリティ
# ──────────────────────────────────────────────────────────────────────────────

def _fmt_sec(sec: float) -> str:
    """秒数を mm:ss.s 形式に変換する。"""
    m = int(sec) // 60
    s = sec % 60
    return f"{m:02d}:{s:04.1f}"


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