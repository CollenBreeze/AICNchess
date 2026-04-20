# DDQN 首版骨架

把 `rl/` 目录放到你项目的 `src/` 下。

## 关键设计
- 动作空间固定为 `90 * 90 = 8100`
- 复用现有 `Board / Move / generate_legal_moves`
- 三次重复仅在 AI / 训练逻辑内作为和棋处理
- **单网络自对弈时，DDQN 目标要做“零和符号翻转”**：
  - `target = reward - gamma * next_q`
  - 因为下一状态轮到对手走

## 第一步怎么跑
在项目根目录执行：

```bash
python src/rl/train_ddqn.py --episodes 200 --teacher-prob 0.5 --teacher-depth 2
```

## 推理怎么接
后面在 `play.py` / `ctai.py` 里可以这样用：

```python
from rl.ddqn_player import choose_ddqn_move

move = choose_ddqn_move(board, "checkpoints_ddqn/ddqn_final.pt")
```

## 建议训练路线
1. 先用 `teacher-prob=0.3~0.5` 混入现有启发式引擎走法，解决冷启动
2. 跑通后把 teacher 概率慢慢降到 0
3. 再加入残局集、开局随机化、评测脚本

默认网络规模已经压到适合 CPU 起步的档位：channels=64, blocks=2, hidden_dim=256。
