# self_play_tuner.py

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime

from board import Board
from engine import EngineParams, PRESET_PARAMS, XiangqiEngine, params_from_preset
from legal_moves import generate_legal_moves

MUTATION_RULES = {
    "rook_value": (20, 800, 1100),
    "knight_value": (16, 350, 650),
    "cannon_value": (16, 380, 650),
    "elephant_value": (10, 150, 320),
    "advisor_value": (10, 150, 320),
    "pawn_value": (10, 80, 220),
    "center_weight": (2, 0, 16),
    "major_advance_weight": (1, 0, 8),
    "pawn_advance_weight": (1, 0, 10),
    "pawn_cross_bonus": (6, 0, 100),
    "king_guard_weight": (2, 0, 40),
    "check_penalty": (8, 40, 220),
    "tempo_bonus": (2, 0, 32),
    "max_quiescence_depth": (1, 2, 14),
}


@dataclass(slots=True)
class MatchResult:
    score_a: float
    score_b: float
    wins_a: int
    wins_b: int
    draws: int
    games: int

    def to_dict(self):
        return asdict(self)


def clamp(value, low, high):
    return max(low, min(high, value))


def mutate_params(params, rng, scale=1.0):
    data = params.to_dict()
    keys = list(MUTATION_RULES)
    rng.shuffle(keys)
    mutate_count = rng.randint(3, min(6, len(keys)))

    for key in keys[:mutate_count]:
        step, low, high = MUTATION_RULES[key]
        delta = rng.randint(-step, step)
        value = data[key] + int(round(delta * scale))
        data[key] = clamp(value, low, high)

    data["name"] = f"{params.name}_mut"
    return EngineParams.from_dict(data)


def adjudicate_by_material(board):
    piece_values = PRESET_PARAMS["balanced"]
    score = 0
    for piece in board.board:
        if piece == 0:
            continue
        abs_piece = abs(piece)
        if abs_piece == 1:
            value = piece_values.king_value
        elif abs_piece == 2:
            value = piece_values.rook_value
        elif abs_piece == 3:
            value = piece_values.knight_value
        elif abs_piece == 4:
            value = piece_values.cannon_value
        elif abs_piece == 5:
            value = piece_values.elephant_value
        elif abs_piece == 6:
            value = piece_values.advisor_value
        else:
            value = piece_values.pawn_value
        score += value if piece > 0 else -value

    if score > 150:
        return 1
    if score < -150:
        return -1
    return 0


def apply_random_opening(board, rng, opening_plies):
    for _ in range(opening_plies):
        moves = generate_legal_moves(board)
        if not moves:
            return
        board.make_move(rng.choice(moves))


def play_game(engine_red, engine_black, depth_red, depth_black, seed, opening_plies=4, max_plies=160):
    rng = random.Random(seed)
    board = Board()
    board.init_startpos()
    apply_random_opening(board, rng, opening_plies)

    if board.is_draw_by_repetition():
        return 0

    for _ in range(max_plies):
        if board.king_pos[1] == -1:
            return -1
        if board.king_pos[-1] == -1:
            return 1

        if board.side > 0:
            move = engine_red.choose_move(board, depth=depth_red)
        else:
            move = engine_black.choose_move(board, depth=depth_black)

        if move is None:
            return -1 if board.side > 0 else 1

        board.make_move(move)

        if board.king_pos[1] == -1:
            return -1
        if board.king_pos[-1] == -1:
            return 1

        if board.is_draw_by_repetition():
            return 0

    return adjudicate_by_material(board)


def play_match(params_a, params_b, games, depth_a, depth_b, opening_plies, seed):
    engine_a = XiangqiEngine(params=params_a)
    engine_b = XiangqiEngine(params=params_b)

    score_a = 0.0
    score_b = 0.0
    wins_a = 0
    wins_b = 0
    draws = 0

    for game_index in range(games):
        game_seed = seed + game_index * 9973

        if game_index % 2 == 0:
            result = play_game(
                engine_red=engine_a,
                engine_black=engine_b,
                depth_red=depth_a,
                depth_black=depth_b,
                seed=game_seed,
                opening_plies=opening_plies,
            )
            if result > 0:
                score_a += 1.0
                wins_a += 1
            elif result < 0:
                score_b += 1.0
                wins_b += 1
            else:
                score_a += 0.5
                score_b += 0.5
                draws += 1
        else:
            result = play_game(
                engine_red=engine_b,
                engine_black=engine_a,
                depth_red=depth_b,
                depth_black=depth_a,
                seed=game_seed,
                opening_plies=opening_plies,
            )
            if result > 0:
                score_b += 1.0
                wins_b += 1
            elif result < 0:
                score_a += 1.0
                wins_a += 1
            else:
                score_a += 0.5
                score_b += 0.5
                draws += 1

    return MatchResult(
        score_a=score_a,
        score_b=score_b,
        wins_a=wins_a,
        wins_b=wins_b,
        draws=draws,
        games=games,
    )


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)


def tune_duel(
    preset_a="aggressive",
    preset_b="solid",
    rounds=8,
    games=4,
    depth_a=2,
    depth_b=2,
    opening_plies=4,
    scale=1.0,
    seed=20260417,
    output_dir=None,
):
    rng = random.Random(seed)
    params_a = params_from_preset(preset_a)
    params_b = params_from_preset(preset_b)

    baseline = play_match(
        params_a=params_a,
        params_b=params_b,
        games=games,
        depth_a=depth_a,
        depth_b=depth_b,
        opening_plies=opening_plies,
        seed=seed,
    )

    logs = [
        {
            "round": 0,
            "accepted_a": False,
            "accepted_b": False,
            "result": baseline.to_dict(),
            "params_a": params_a.to_dict(),
            "params_b": params_b.to_dict(),
        }
    ]

    print(
        f"[round 0] baseline  A={baseline.score_a:.1f}  B={baseline.score_b:.1f}  "
        f"W/L/D={baseline.wins_a}/{baseline.wins_b}/{baseline.draws}"
    )

    for round_index in range(1, rounds + 1):
        accepted_a = False
        accepted_b = False

        candidate_a = mutate_params(params_a, rng, scale=scale)
        result_a = play_match(
            params_a=candidate_a,
            params_b=params_b,
            games=games,
            depth_a=depth_a,
            depth_b=depth_b,
            opening_plies=opening_plies,
            seed=seed + round_index * 1000 + 1,
        )
        if result_a.score_a > baseline.score_a:
            params_a = candidate_a
            baseline = result_a
            accepted_a = True

        candidate_b = mutate_params(params_b, rng, scale=scale)
        result_b = play_match(
            params_a=params_a,
            params_b=candidate_b,
            games=games,
            depth_a=depth_a,
            depth_b=depth_b,
            opening_plies=opening_plies,
            seed=seed + round_index * 1000 + 2,
        )
        if result_b.score_b > baseline.score_b:
            params_b = candidate_b
            baseline = result_b
            accepted_b = True

        logs.append(
            {
                "round": round_index,
                "accepted_a": accepted_a,
                "accepted_b": accepted_b,
                "result": baseline.to_dict(),
                "params_a": params_a.to_dict(),
                "params_b": params_b.to_dict(),
            }
        )

        print(
            f"[round {round_index}] A({'+' if accepted_a else '-'}) B({'+' if accepted_b else '-'})  "
            f"A={baseline.score_a:.1f}  B={baseline.score_b:.1f}  "
            f"W/L/D={baseline.wins_a}/{baseline.wins_b}/{baseline.draws}"
        )

    saved = {}
    if output_dir is not None:
        ensure_output_dir(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path_a = os.path.join(output_dir, f"params_A_{timestamp}.json")
        path_b = os.path.join(output_dir, f"params_B_{timestamp}.json")
        path_log = os.path.join(output_dir, f"tuning_log_{timestamp}.json")

        with open(path_a, "w", encoding="utf-8") as f:
            json.dump(params_a.to_dict(), f, ensure_ascii=False, indent=2)
        with open(path_b, "w", encoding="utf-8") as f:
            json.dump(params_b.to_dict(), f, ensure_ascii=False, indent=2)
        with open(path_log, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)

        saved = {"params_a": path_a, "params_b": path_b, "log": path_log}
        print(f"保存完成: {saved}")

    return {
        "params_a": params_a,
        "params_b": params_b,
        "baseline": baseline,
        "logs": logs,
        "saved": saved,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="中国象棋双 AI 自对弈调参器")
    parser.add_argument("--preset-a", default="aggressive", choices=list(PRESET_PARAMS))
    parser.add_argument("--preset-b", default="solid", choices=list(PRESET_PARAMS))
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--depth-a", type=int, default=2)
    parser.add_argument("--depth-b", type=int, default=2)
    parser.add_argument("--opening-plies", type=int, default=4)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260417)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tune_duel(
        preset_a=args.preset_a,
        preset_b=args.preset_b,
        rounds=args.rounds,
        games=args.games,
        depth_a=args.depth_a,
        depth_b=args.depth_b,
        opening_plies=args.opening_plies,
        scale=args.scale,
        seed=args.seed,
        output_dir=args.output_dir,
    )
