# train_nnue_gui.py

from __future__ import annotations

import argparse
import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = Path(__file__).resolve().with_name("train_nnue.py")
DEFAULT_OUTPUT = PROJECT_ROOT / "checkpoints_nnue" / "xiangqi_nnue_v2.npz"

DATASET_RE = re.compile(r"\[dataset\]\s+progress=(\d+)/(\d+)")
EPOCH_RE = re.compile(r"\[epoch\]\s+(\d+)/(\d+)\s+loss=([0-9.]+)")
DONE_RE = re.compile(r"\[done\]\s+best_epoch=(\d+)\s+best_loss=([0-9.]+)")



def read_nnue_shapes(path: str | os.PathLike[str]):
    try:
        import numpy as np

        data = np.load(path, allow_pickle=False)
        acc_size = int(data["ft_weight"].shape[0])
        hidden_size = int(data["l1_weight"].shape[0])
        return acc_size, hidden_size
    except Exception:
        return None


class TrainNNUEGUI(tk.Tk):
    def __init__(self, init_weights: str = "", output_path: str = ""):
        super().__init__()
        self.title("Xiangqi NNUE 训练器")
        self.geometry("980x760")
        self.minsize(900, 680)

        self.process: subprocess.Popen[str] | None = None
        self.reader_thread: threading.Thread | None = None
        self.log_queue: queue.Queue[tuple[str, str]] = queue.Queue()

        self.samples_var = tk.StringVar(value="1200")
        self.teacher_depth_var = tk.StringVar(value="2")
        self.teacher_preset_var = tk.StringVar(value="balanced")
        self.guide_preset_var = tk.StringVar(value="balanced")
        self.guide_depth_var = tk.StringVar(value="1")
        self.max_random_plies_var = tk.StringVar(value="24")
        self.guide_prob_var = tk.StringVar(value="0.35")
        self.epochs_var = tk.StringVar(value="6")
        self.batch_size_var = tk.StringVar(value="64")
        self.lr_var = tk.StringVar(value="0.002")
        self.acc_size_var = tk.StringVar(value="64")
        self.hidden_size_var = tk.StringVar(value="32")
        self.device_var = tk.StringVar(value="auto")
        self.seed_var = tk.StringVar(value="2026")
        self.init_weights_var = tk.StringVar(value=init_weights or "")
        self.output_var = tk.StringVar(value=output_path or str(DEFAULT_OUTPUT))
        self.status_var = tk.StringVar(value="准备就绪")
        self.progress_var = tk.StringVar(value="等待开始")

        self._build_ui()
        self._maybe_fill_shapes_from_init()
        self.after(120, self._poll_log_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        root = ttk.Frame(self, padding=12)
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(2, weight=1)

        form = ttk.LabelFrame(root, text="训练参数", padding=10)
        form.grid(row=0, column=0, sticky="nsew")
        for col in range(4):
            form.columnconfigure(col, weight=1)

        self._add_entry(form, 0, 0, "样本数", self.samples_var)
        self._add_entry(form, 0, 2, "教师深度", self.teacher_depth_var)
        self._add_combo(form, 1, 0, "教师预设", self.teacher_preset_var, ["balanced", "aggressive", "solid"])
        self._add_combo(form, 1, 2, "引导预设", self.guide_preset_var, ["balanced", "aggressive", "solid"])
        self._add_entry(form, 2, 0, "引导深度", self.guide_depth_var)
        self._add_entry(form, 2, 2, "最大随机步数", self.max_random_plies_var)
        self._add_entry(form, 3, 0, "引导概率", self.guide_prob_var)
        self._add_entry(form, 3, 2, "训练轮数", self.epochs_var)
        self._add_entry(form, 4, 0, "批大小", self.batch_size_var)
        self._add_entry(form, 4, 2, "学习率", self.lr_var)
        self._add_entry(form, 5, 0, "Accumulator大小", self.acc_size_var)
        self._add_entry(form, 5, 2, "Hidden大小", self.hidden_size_var)
        self._add_combo(form, 6, 0, "设备", self.device_var, ["auto", "cpu", "cuda", "mps"])
        self._add_entry(form, 6, 2, "随机种子", self.seed_var)

        path_frame = ttk.LabelFrame(root, text="文件", padding=10)
        path_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        path_frame.columnconfigure(1, weight=1)

        ttk.Label(path_frame, text="初始权重").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(path_frame, textvariable=self.init_weights_var).grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Button(path_frame, text="浏览", command=self._browse_init_weights).grid(row=0, column=2, padx=(8, 0), pady=4)

        ttk.Label(path_frame, text="输出权重").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(path_frame, textvariable=self.output_var).grid(row=1, column=1, sticky="ew", pady=4)
        ttk.Button(path_frame, text="浏览", command=self._browse_output).grid(row=1, column=2, padx=(8, 0), pady=4)

        controls = ttk.Frame(root)
        controls.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        controls.columnconfigure(0, weight=1)
        controls.rowconfigure(2, weight=1)

        button_bar = ttk.Frame(controls)
        button_bar.grid(row=0, column=0, sticky="ew")
        ttk.Button(button_bar, text="开始训练", command=self.start_training).pack(side=tk.LEFT)
        ttk.Button(button_bar, text="停止训练", command=self.stop_training).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(button_bar, text="打开输出目录", command=self._open_output_dir).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(button_bar, text="从初始权重读取结构", command=self._maybe_fill_shapes_from_init).pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(controls, textvariable=self.status_var).grid(row=1, column=0, sticky="w", pady=(10, 2))
        self.progress = ttk.Progressbar(controls, mode="indeterminate")
        self.progress.grid(row=1, column=0, sticky="e", pady=(10, 2))
        ttk.Label(controls, textvariable=self.progress_var).grid(row=1, column=0, sticky="e", padx=(0, 170), pady=(10, 2))

        self.log_text = scrolledtext.ScrolledText(controls, wrap=tk.WORD, font=("Consolas", 10))
        self.log_text.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        self.log_text.configure(state=tk.DISABLED)

    def _add_entry(self, parent, row, col, label, variable):
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=col + 1, sticky="ew", pady=4)

    def _add_combo(self, parent, row, col, label, variable, values):
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", padx=(0, 8), pady=4)
        combo = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly")
        combo.grid(row=row, column=col + 1, sticky="ew", pady=4)

    def _append_log(self, text: str):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _browse_init_weights(self):
        path = filedialog.askopenfilename(
            title="选择初始 NNUE 权重",
            filetypes=[("NNUE weights", "*.npz"), ("All files", "*.*")],
            initialdir=str(PROJECT_ROOT / "checkpoints_nnue"),
        )
        if path:
            self.init_weights_var.set(path)
            self._maybe_fill_shapes_from_init()

    def _browse_output(self):
        path = filedialog.asksaveasfilename(
            title="选择输出权重路径",
            defaultextension=".npz",
            filetypes=[("NNUE weights", "*.npz"), ("All files", "*.*")],
            initialdir=str(PROJECT_ROOT / "checkpoints_nnue"),
            initialfile=Path(self.output_var.get() or DEFAULT_OUTPUT.name).name,
        )
        if path:
            self.output_var.set(path)

    def _open_output_dir(self):
        output_path = Path(self.output_var.get()).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        target = str(output_path.parent)
        try:
            if sys.platform.startswith("win"):
                os.startfile(target)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", target])
            else:
                subprocess.Popen(["xdg-open", target])
        except Exception as exc:
            messagebox.showerror("打开目录失败", str(exc))

    def _maybe_fill_shapes_from_init(self):
        path = self.init_weights_var.get().strip()
        if not path or not os.path.isfile(path):
            return
        shapes = read_nnue_shapes(path)
        if not shapes:
            messagebox.showwarning("读取失败", "没能从初始权重读取网络结构")
            return
        acc_size, hidden_size = shapes
        self.acc_size_var.set(str(acc_size))
        self.hidden_size_var.set(str(hidden_size))
        self.status_var.set(f"已从初始权重读取结构：acc={acc_size}, hidden={hidden_size}")

    def _build_command(self):
        cmd = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--samples", self.samples_var.get().strip(),
            "--teacher-depth", self.teacher_depth_var.get().strip(),
            "--teacher-preset", self.teacher_preset_var.get().strip(),
            "--guide-preset", self.guide_preset_var.get().strip(),
            "--guide-depth", self.guide_depth_var.get().strip(),
            "--max-random-plies", self.max_random_plies_var.get().strip(),
            "--guide-prob", self.guide_prob_var.get().strip(),
            "--epochs", self.epochs_var.get().strip(),
            "--batch-size", self.batch_size_var.get().strip(),
            "--lr", self.lr_var.get().strip(),
            "--acc-size", self.acc_size_var.get().strip(),
            "--hidden-size", self.hidden_size_var.get().strip(),
            "--device", self.device_var.get().strip(),
            "--seed", self.seed_var.get().strip(),
            "--output", self.output_var.get().strip(),
        ]
        init_weights = self.init_weights_var.get().strip()
        if init_weights:
            cmd.extend(["--init-weights", init_weights])
        return cmd

    def start_training(self):
        if self.process is not None:
            messagebox.showinfo("训练中", "当前已有训练任务在运行")
            return

        if not TRAIN_SCRIPT.is_file():
            messagebox.showerror("缺少脚本", f"未找到训练脚本：{TRAIN_SCRIPT}")
            return

        output_path = Path(self.output_var.get().strip()).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            cmd = self._build_command()
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return

        self._append_log("\n>>> " + " ".join(cmd) + "\n")
        self.status_var.set("训练已启动")
        self.progress_var.set("准备训练…")
        self.progress.start(10)

        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        except Exception as exc:
            self.progress.stop()
            self.process = None
            messagebox.showerror("启动失败", str(exc))
            return

        self.reader_thread = threading.Thread(target=self._reader_worker, daemon=True)
        self.reader_thread.start()

    def _reader_worker(self):
        assert self.process is not None
        assert self.process.stdout is not None
        try:
            for line in self.process.stdout:
                self.log_queue.put(("log", line))
        finally:
            return_code = self.process.wait()
            self.log_queue.put(("done", str(return_code)))

    def stop_training(self):
        if self.process is None:
            return
        try:
            self.process.terminate()
        except Exception:
            pass
        self.status_var.set("正在停止训练…")

    def _handle_log_line(self, line: str):
        self._append_log(line)
        line = line.strip()
        if not line:
            return

        m = DATASET_RE.search(line)
        if m:
            self.progress_var.set(f"采样 {m.group(1)}/{m.group(2)}")
            self.status_var.set("正在构建训练集")
            return

        m = EPOCH_RE.search(line)
        if m:
            self.progress_var.set(f"Epoch {m.group(1)}/{m.group(2)} · loss={m.group(3)}")
            self.status_var.set("正在训练网络")
            return

        m = DONE_RE.search(line)
        if m:
            self.progress_var.set(f"最佳轮次 {m.group(1)} · best_loss={m.group(2)}")
            self.status_var.set("训练完成")
            return

        if line.startswith("saved:"):
            self.status_var.set("权重已保存")
            return

    def _poll_log_queue(self):
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "log":
                    self._handle_log_line(payload)
                elif kind == "done":
                    self.progress.stop()
                    if payload == "0":
                        self.status_var.set("训练进程已结束")
                    else:
                        self.status_var.set(f"训练进程退出，返回码 {payload}")
                    self.process = None
                    self.reader_thread = None
        except queue.Empty:
            pass
        self.after(120, self._poll_log_queue)

    def _on_close(self):
        if self.process is not None:
            if not messagebox.askyesno("退出", "训练仍在运行，确定要退出并终止训练吗？"):
                return
            self.stop_training()
        self.destroy()



def parse_args():
    parser = argparse.ArgumentParser(description="NNUE 训练图形界面")
    parser.add_argument("--init-weights", default="")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = TrainNNUEGUI(init_weights=args.init_weights, output_path=args.output)
    app.mainloop()
