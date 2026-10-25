import numpy as np
import torch

from chess_rl.encoding import POLICY_SIZE
from chess_rl.mcts import MCTSConfig
from chess_rl.model import ChessNet, ModelConfig, NetworkEvaluator, load_checkpoint, save_checkpoint
from chess_rl.selfplay import ReplayBuffer, SelfPlayConfig, play_games
from chess_rl.train import main as train_main


def tiny_model():
    return ChessNet(ModelConfig(channels=8, blocks=1, value_hidden=16))


def test_model_shapes():
    model = tiny_model()
    logits, value = model(torch.zeros(3, 19, 8, 8))
    assert logits.shape == (3, POLICY_SIZE)
    assert value.shape == (3,)


def test_selfplay_produces_consistent_samples():
    cfg = SelfPlayConfig(games_per_batch=2, max_plies=20, mcts=MCTSConfig(num_simulations=4))
    records = play_games(NetworkEvaluator(tiny_model()), 3, cfg, np.random.default_rng(0))
    assert len(records) == 3
    buffer = ReplayBuffer(capacity=25)
    for r in records:
        assert len(r.samples) == len(r.moves)
        for s in r.samples:
            assert np.isclose(s.policy_probs.sum(), 1.0, atol=1e-5)
            assert s.value in (-1.0, 0.0, 1.0)
        buffer.add_game(r)
    assert len(buffer) == 25  # capacity respected
    assert len(buffer.sample_batch(8)) == 8


def test_checkpoint_roundtrip(tmp_path):
    model = tiny_model()
    path = tmp_path / "m.pt"
    save_checkpoint(str(path), model, iteration=3)
    loaded, payload = load_checkpoint(str(path))
    assert payload["iteration"] == 3
    x = torch.randn(2, 19, 8, 8)
    model.eval(), loaded.eval()
    torch.testing.assert_close(model(x)[0], loaded(x)[0])


def test_training_loop_smoke(tmp_path):
    train_main([
        "--run-dir", str(tmp_path), "--iterations", "2", "--device", "cpu",
        "--channels", "8", "--blocks", "1", "--games-per-iter", "2", "--parallel-games", "2",
        "--simulations", "4", "--max-plies", "16", "--min-buffer", "1", "--train-steps", "2",
        "--batch-size", "8", "--eval-every", "2", "--eval-games", "2", "--eval-simulations", "2",
    ])
    assert (tmp_path / "latest.pt").exists()
    assert len((tmp_path / "metrics.jsonl").read_text().splitlines()) == 2
