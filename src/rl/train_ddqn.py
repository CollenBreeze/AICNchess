from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ is None or __package__ == "":
    SRC_DIR = Path(__file__).resolve().parents[1]
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    from ai_heuristic import choose_heuristic_move
    from rl.action_space import legal_action_ids, move_to_action
    from rl.ddqn_agent import DDQNAgent, DDQNConfig
    from rl.replay_buffer import ReplayBuffer
    from rl.state_encoder import encode_snapshot
    from rl.xiangqi_env import XiangqiEnv
else:
    from ai_heuristic import choose_heuristic_move
    from .action_space import legal_action_ids, move_to_action
    from .ddqn_agent import DDQNAgent, DDQNConfig
    from .replay_buffer import ReplayBuffer
    from .state_encoder import encode_snapshot
    from .xiangqi_env import XiangqiEnv


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DDQN Xiangqi agent")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-buffer-size", type=int, default=2000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--target-sync", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--out-dir", type=str, default="checkpoints_ddqn")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher-prob", type=float, default=0.25)
    parser.add_argument("--teacher-depth", type=int, default=2)
    parser.add_argument("--teacher-preset", type=str, default="balanced")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    return parser.parse_args()


def maybe_teacher_action(env, teacher_prob: float, teacher_depth: int, teacher_preset: str):
    if teacher_prob <= 0.0 or random.random() > teacher_prob:
        return None

    teacher_move = choose_heuristic_move(
        env.board,
        depth=teacher_depth,
        preset=teacher_preset,
        params_path=None,
    )
    if teacher_move is None:
        return None
    return move_to_action(teacher_move)


def main():
    args = parse_args()
    set_seed(args.seed)

    env = XiangqiEnv(max_plies=args.max_plies)
    config = DDQNConfig(
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        min_buffer_size=args.min_buffer_size,
        target_sync_interval=args.target_sync,
        channels=args.channels,
        num_blocks=args.blocks,
        hidden_dim=args.hidden_dim,
    )
    agent = DDQNAgent(config=config, device=args.device)
    buffer = ReplayBuffer(capacity=args.buffer_size)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for episode in range(1, args.episodes + 1):
        state_snapshot = env.reset()
        done = False
        episode_reward = 0.0
        episode_plies = 0
        last_metrics = None
        final_info = {"reason": "unknown"}

        while not done:
            legal_actions = env.legal_action_ids()
            if not legal_actions:
                break

            teacher_action = maybe_teacher_action(
                env,
                teacher_prob=args.teacher_prob,
                teacher_depth=args.teacher_depth,
                teacher_preset=args.teacher_preset,
            )

            if teacher_action is not None and teacher_action in legal_actions:
                action = teacher_action
            else:
                encoded_state = encode_snapshot(*state_snapshot)
                action = agent.select_action(encoded_state, legal_actions, explore=True)

            next_state_snapshot, reward, done, info = env.step(action)
            next_legal_actions = [] if done else env.legal_action_ids()

            buffer.push(
                state_board=state_snapshot[0],
                state_side=state_snapshot[1],
                state_repeat=state_snapshot[2],
                action=action,
                reward=reward,
                next_board=next_state_snapshot[0],
                next_side=next_state_snapshot[1],
                next_repeat=next_state_snapshot[2],
                done=done,
                next_legal_actions=next_legal_actions,
            )

            last_metrics = agent.train_step(buffer)
            episode_reward += reward
            episode_plies += 1
            state_snapshot = next_state_snapshot
            final_info = info

        if episode % 10 == 0 or episode == 1:
            metrics_text = ""
            if last_metrics is not None:
                metrics_text = (
                    f" loss={last_metrics['loss']:.4f}"
                    f" eps={last_metrics['epsilon']:.3f}"
                    f" q={last_metrics['q_mean']:.3f}"
                )
            print(
                f"[episode {episode:05d}]"
                f" plies={episode_plies:03d}"
                f" reward={episode_reward:+.2f}"
                f" reason={final_info.get('reason')}{metrics_text}"
            )

        if episode % args.save_every == 0:
            agent.save(out_dir / f"ddqn_ep{episode:05d}.pt")

    final_path = out_dir / "ddqn_final.pt"
    agent.save(final_path)
    print(f"saved model to {final_path}")


if __name__ == "__main__":
    main()
