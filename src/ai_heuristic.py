# ai_heuristic.py

"""
兼容层。

原来的 ai_heuristic.py 里把大量“局面评估 / 着法排序 / 搜索”混在一起，
而且排序阶段会反复调用 generate_legal_moves，速度非常慢。

现在保留 choose_heuristic_move 这个老接口，底层直接转发到新的引擎实现，
这样 ctai.py 等旧代码不需要大改也能直接提速。
"""

from engine import XiangqiEngine, create_engine

_ENGINE_CACHE = {}


def get_engine(preset="balanced", params_path=None):
    key = (preset, params_path)
    engine = _ENGINE_CACHE.get(key)
    if engine is None:
        engine = create_engine(preset=preset)
        if params_path:
            engine.load_params(params_path)
        _ENGINE_CACHE[key] = engine
    return engine


def choose_heuristic_move(board, depth=3, preset="balanced", params_path=None):
    return get_engine(preset=preset, params_path=params_path).choose_move(board, depth=depth)


def build_custom_engine(params_path, preset="balanced"):
    engine = XiangqiEngine()
    if preset:
        engine = create_engine(preset=preset)
    engine.load_params(params_path)
    return engine
