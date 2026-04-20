# nnue_player.py

from __future__ import annotations

import os
from pathlib import Path

from nnue_engine import create_nnue_engine

_ENGINE_CACHE = {}



def _engine_cache_key(weights_path: str, preset: str) -> tuple[str, str, float | None]:
    resolved = str(Path(weights_path).resolve())
    try:
        mtime = os.path.getmtime(resolved)
    except OSError:
        mtime = None
    return resolved, str(preset), mtime



def clear_nnue_engine_cache():
    _ENGINE_CACHE.clear()



def load_nnue_engine(weights_path: str, preset: str = "balanced"):
    key = _engine_cache_key(weights_path, preset)
    engine = _ENGINE_CACHE.get(key)
    if engine is None:
        if len(_ENGINE_CACHE) >= 8:
            _ENGINE_CACHE.clear()
        engine = create_nnue_engine(weights_path, preset=preset)
        _ENGINE_CACHE[key] = engine
    return engine



def choose_nnue_move(board, depth=4, weights_path=None, preset="balanced"):
    if not weights_path:
        raise ValueError("NNUE 权重路径不能为空")
    engine = load_nnue_engine(weights_path, preset=preset)
    return engine.choose_move(board, depth=depth)
