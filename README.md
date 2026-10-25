# chess-rl

Learning chess from scratch with reinforcement learning, in the style of
[AlphaZero](https://arxiv.org/abs/1712.01815): a neural network plays games
against itself, guided by Monte Carlo Tree Search (MCTS), and learns from the
outcomes. No human games or handcrafted evaluation are used. Only the rules,
via [python-chess](https://python-chess.readthedocs.io/).

## How it works

```
        ┌─────────────────────────────────────────────┐
        │                                             │
        ▼                                             │
  ┌───────────┐  games + MCTS visit counts  ┌──────────────┐
  │ Self-play │ ──────────────────────────▶ │ Replay buffer│
  │ (MCTS+NN) │                             └──────┬───────┘
  └───────────┘                                    │ random batches
        ▲                                          ▼
        │        updated weights            ┌──────────────┐
        └────────────────────────────────── │   Training   │
                                            └──────────────┘
```

1. **Self-play** (`chess_rl/selfplay.py`): the current network plays games
   against itself. At every move MCTS runs N simulations; the network's
   policy head provides move priors and its value head scores leaf positions.
   Many games run concurrently so their leaf evaluations share one batched
   forward pass.
2. **Training targets**: for each position we store the MCTS visit
   distribution (a stronger policy than the raw network) and the final game
   result from that player's perspective.
3. **Training** (`chess_rl/train.py`): the network is trained to match the
   visit distribution (cross-entropy) and predict the result (MSE).
4. Repeat. Periodically the model plays a random mover and its previous
   checkpoint, and the Elo difference is logged.

| Module | Contents |
| --- | --- |
| `encoding.py` | 19×8×8 board planes (side-to-move relative) and the AlphaZero 8×8×73 = 4672 move encoding |
| `model.py` | ResNet with policy and value heads, batched evaluator, checkpoint helpers |
| `mcts.py` | PUCT search with Dirichlet root noise, batched across trees |
| `selfplay.py` | Concurrent self-play games and the replay buffer |
| `train.py` | The main RL loop |
| `evaluate.py` | Matches between checkpoints and/or a random player |
| `play.py` | Play against a checkpoint in the terminal |
| `uci.py` | UCI wrapper for chess GUIs and lichess-bot |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Usage

Train (defaults are sized for a laptop CPU; scale up on a GPU):

```bash
python -m chess_rl.train --run-dir runs/first --iterations 100
# GPU example: bigger net, more search, more games
python -m chess_rl.train --run-dir runs/big --channels 128 --blocks 10 \
    --simulations 200 --games-per-iter 256 --parallel-games 128 --train-steps 1000
# resume
python -m chess_rl.train --run-dir runs/first --resume --iterations 50
```

Every iteration appends a JSON line to `runs/<name>/metrics.jsonl` with
losses, game lengths, results and evaluation scores.

Evaluate and play:

```bash
python -m chess_rl.evaluate runs/first/latest.pt random --games 20
python -m chess_rl.evaluate runs/first/latest.pt runs/first/iter_0010.pt
python -m chess_rl.play runs/first/latest.pt --color white
python -m chess_rl.uci runs/first/latest.pt     # point a GUI at this command
```

## Realistic expectations

AlphaZero used thousands of TPUs for 44 million games. On a single CPU this
project will learn slowly. Expect it to beat a random mover after a modest
amount of training, and to need a GPU and days of self-play for real
strength. Ways to go faster, roughly in order of payoff:

- **Use a GPU** and increase `--parallel-games` so batches stay large.
- **Bootstrap from supervised learning** on human/engine games (e.g. a
  Lichess PGN dump), then continue with self-play. Not implemented yet.
- **Reuse the search tree** between moves, and run self-play in multiple
  processes.
- **Resign/adjudicate** clearly decided games early to save compute.
