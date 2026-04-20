# train_nnue.py

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from board import Board
from engine import create_engine
from legal_moves import generate_legal_moves
from nnue_model import NUM_FEATURES, XiangqiNNUE, dense_pair_from_board


class TorchNNUE(nn.Module):
    def __init__(self, num_features=NUM_FEATURES, acc_size=64, hidden_size=32):
        super().__init__()
        self.ft = nn.Linear(num_features, acc_size)
        self.psqt = nn.Linear(num_features, 1)
        self.l1 = nn.Linear(acc_size * 2, hidden_size)
        self.l2 = nn.Linear(hidden_size, 1)

    def forward(self, red_x, black_x, side):
        red_acc = torch.clamp(self.ft(red_x), 0.0, 1.0)
        black_acc = torch.clamp(self.ft(black_x), 0.0, 1.0)
        red_psqt = self.psqt(red_x)
        black_psqt = self.psqt(black_x)

        white_first = torch.cat([red_acc, black_acc], dim=1)
        black_first = torch.cat([black_acc, red_acc], dim=1)
        side = side.view(-1, 1)
        hidden_input = torch.where(side > 0, white_first, black_first)
        psqt_term = torch.where(side > 0, red_psqt - black_psqt, black_psqt - red_psqt)
        hidden = torch.clamp(self.l1(hidden_input), 0.0, 1.0)
        return self.l2(hidden) + psqt_term


@dataclass(slots=True)
class Sample:
    red: np.ndarray
    black: np.ndarray
    side: int
    residual: float



def log(message: str):
    print(message, flush=True)



def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def resolve_device(device_name: str | None):
    name = str(device_name or "auto").strip().lower()
    if name in ("", "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("请求了 cuda，但当前环境没有可用 CUDA 设备")
        return torch.device("cuda")
    if name == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("请求了 mps，但当前环境没有可用的 Apple Metal 设备")
        return torch.device("mps")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"未知 device: {device_name}")



def load_initial_weights(model: TorchNNUE, weights_path: str):
    nnue = XiangqiNNUE.load(weights_path)
    if model.ft.weight.shape != nnue.ft_weight.shape:
        raise ValueError(
            f"初始权重 acc_size 不匹配: 训练模型 {tuple(model.ft.weight.shape)} vs 权重 {tuple(nnue.ft_weight.shape)}"
        )
    if model.l1.weight.shape != nnue.l1_weight.shape:
        raise ValueError(
            f"初始权重 hidden_size 不匹配: 训练模型 {tuple(model.l1.weight.shape)} vs 权重 {tuple(nnue.l1_weight.shape)}"
        )

    with torch.no_grad():
        model.ft.weight.copy_(torch.from_numpy(nnue.ft_weight))
        model.ft.bias.copy_(torch.from_numpy(nnue.ft_bias))
        model.psqt.weight.copy_(torch.from_numpy(nnue.psqt_weight.reshape(1, -1)))
        model.psqt.bias.fill_(float(nnue.psqt_bias))
        model.l1.weight.copy_(torch.from_numpy(nnue.l1_weight))
        model.l1.bias.copy_(torch.from_numpy(nnue.l1_bias))
        model.l2.weight.copy_(torch.from_numpy(nnue.l2_weight))
        model.l2.bias.fill_(float(nnue.l2_bias))



def sample_position(rng: random.Random, guide_engine, max_random_plies: int, guide_prob: float, guide_depth: int):
    board = Board()
    board.init_startpos()
    plies = rng.randint(0, max_random_plies)

    for _ in range(plies):
        moves = generate_legal_moves(board)
        if not moves:
            break
        if rng.random() < guide_prob:
            move = guide_engine.choose_move(board, depth=guide_depth)
            if move is None:
                break
        else:
            move = rng.choice(moves)
        board.make_move(move)
        if board.is_draw_by_repetition():
            break
    return board



def build_dataset(
    sample_count: int,
    teacher_depth: int,
    teacher_preset: str,
    guide_preset: str,
    max_random_plies: int,
    guide_prob: float,
    guide_depth: int,
    seed: int,
):
    rng = random.Random(seed)
    teacher = create_engine(teacher_preset)
    guide = create_engine(guide_preset)

    samples = []
    seen_hashes = set()
    attempts = 0
    max_attempts = max(sample_count * 6, sample_count + 100)

    while len(samples) < sample_count and attempts < max_attempts:
        attempts += 1
        board = sample_position(
            rng,
            guide,
            max_random_plies=max_random_plies,
            guide_prob=guide_prob,
            guide_depth=guide_depth,
        )

        if board.zhash in seen_hashes:
            continue
        seen_hashes.add(board.zhash)

        moves = generate_legal_moves(board)
        if not moves:
            continue

        base_eval = teacher.evaluate(board)
        teacher_score = teacher.analyze(board, depth=teacher_depth)["score"]
        red, black = dense_pair_from_board(board)
        samples.append(
            Sample(
                red=red.astype(np.uint8, copy=False),
                black=black.astype(np.uint8, copy=False),
                side=board.side,
                residual=float(teacher_score - base_eval),
            )
        )

        if len(samples) % 50 == 0 or len(samples) == sample_count:
            log(f"[dataset] progress={len(samples)}/{sample_count}")

    if len(samples) < sample_count:
        log(f"[dataset] 仅采到 {len(samples)}/{sample_count} 个去重局面，继续使用已采样数据训练")

    if not samples:
        raise RuntimeError("没有采到可用局面")

    red = np.stack([s.red for s in samples]).astype(np.uint8)
    black = np.stack([s.black for s in samples]).astype(np.uint8)
    side = np.asarray([s.side for s in samples], dtype=np.float32)
    residual = np.asarray([s.residual for s in samples], dtype=np.float32)
    return red, black, side, residual



def export_model(model: TorchNNUE, output_path: str | Path):
    nnue = XiangqiNNUE(
        ft_weight=model.ft.weight.detach().cpu().numpy(),
        ft_bias=model.ft.bias.detach().cpu().numpy(),
        psqt_weight=model.psqt.weight.detach().cpu().numpy().reshape(-1),
        psqt_bias=float(model.psqt.bias.detach().cpu().numpy().reshape(())),
        l1_weight=model.l1.weight.detach().cpu().numpy(),
        l1_bias=model.l1.bias.detach().cpu().numpy(),
        l2_weight=model.l2.weight.detach().cpu().numpy(),
        l2_bias=float(model.l2.bias.detach().cpu().numpy().reshape(())),
        version="xiangqi-nnue-acc-v3",
    )
    nnue.save(output_path)



def train(args):
    set_global_seed(args.seed)
    device = resolve_device(args.device)
    log(f"[setup] device={device.type}")
    log(f"[setup] teacher_preset={args.teacher_preset} guide_preset={args.guide_preset}")

    red, black, side, residual = build_dataset(
        sample_count=args.samples,
        teacher_depth=args.teacher_depth,
        teacher_preset=args.teacher_preset,
        guide_preset=args.guide_preset,
        max_random_plies=args.max_random_plies,
        guide_prob=args.guide_prob,
        guide_depth=args.guide_depth,
        seed=args.seed,
    )

    model = TorchNNUE(acc_size=args.acc_size, hidden_size=args.hidden_size).to(device)
    if args.init_weights:
        load_initial_weights(model, args.init_weights)
        log(f"[setup] init_weights={args.init_weights}")

    ds = TensorDataset(
        torch.from_numpy(red),
        torch.from_numpy(black),
        torch.from_numpy(side),
        torch.from_numpy(residual).unsqueeze(1),
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.SmoothL1Loss(beta=16.0)

    best_loss = float("inf")
    best_state = None
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0

        for red_x, black_x, stm, target in loader:
            red_x = red_x.to(device=device, dtype=torch.float32)
            black_x = black_x.to(device=device, dtype=torch.float32)
            stm = stm.to(device=device, dtype=torch.float32)
            target = target.to(device=device, dtype=torch.float32)

            pred = model(red_x, black_x, stm)
            loss = loss_fn(pred, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = red_x.size(0)
            total_loss += float(loss.detach().cpu()) * batch_size
            total_count += batch_size

        mean_loss = total_loss / max(1, total_count)
        log(f"[epoch] {epoch}/{args.epochs} loss={mean_loss:.4f}")

        if mean_loss < best_loss:
            best_loss = mean_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    export_model(model, args.output)
    log(f"[done] best_epoch={best_epoch} best_loss={best_loss:.4f}")
    log(f"saved: {args.output}")



def parse_args():
    parser = argparse.ArgumentParser(description="训练一个给搜索引擎做残差修正的 Xiangqi NNUE")
    parser.add_argument("--samples", type=int, default=1200)
    parser.add_argument("--teacher-depth", type=int, default=2)
    parser.add_argument("--teacher-preset", default="balanced")
    parser.add_argument("--guide-preset", default="balanced")
    parser.add_argument("--guide-depth", type=int, default=1)
    parser.add_argument("--max-random-plies", type=int, default=24)
    parser.add_argument("--guide-prob", type=float, default=0.35)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--acc-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--init-weights", default=None)
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent.parent / "checkpoints_nnue" / "xiangqi_nnue_v2.npz"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
