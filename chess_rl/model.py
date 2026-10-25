"""Policy/value residual network (AlphaZero style)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import NUM_PLANES, POLICY_SIZE, encode_board, legal_move_indices


@dataclass
class ModelConfig:
    channels: int = 64
    blocks: int = 6
    value_hidden: int = 128


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class ChessNet(nn.Module):
    """Outputs policy logits over POLICY_SIZE moves and a value in [-1, 1].

    The value is from the perspective of the side to move.
    """

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config.channels

        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(),
        )
        self.tower = nn.Sequential(*[ResidualBlock(c) for _ in range(self.config.blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(c, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 64, POLICY_SIZE),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(c, 4, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * 64, self.config.value_hidden),
            nn.ReLU(),
            nn.Linear(self.config.value_hidden, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.tower(self.stem(x))
        return self.policy_head(x), self.value_head(x).squeeze(-1)


class NetworkEvaluator:
    """Batched position evaluator used by MCTS.

    `evaluate(boards)` returns, for each board, the legal moves, their prior
    probabilities (softmax restricted to legal moves) and a scalar value for
    the side to move.
    """

    def __init__(self, model: ChessNet, device: torch.device | str = "cpu"):
        self.model = model
        self.device = torch.device(device)

    @torch.no_grad()
    def evaluate(self, boards: list[chess.Board]) -> list[tuple[list[chess.Move], np.ndarray, float]]:
        self.model.eval()
        x = torch.from_numpy(np.stack([encode_board(b) for b in boards])).to(self.device)
        logits, values = self.model(x)
        logits = logits.float().cpu().numpy()
        values = values.float().cpu().numpy()

        results = []
        for board, row, value in zip(boards, logits, values):
            moves, idx = legal_move_indices(board)
            legal_logits = row[idx]
            legal_logits = legal_logits - legal_logits.max()
            priors = np.exp(legal_logits)
            priors /= priors.sum()
            results.append((moves, priors, float(value)))
        return results


def save_checkpoint(path: str, model: ChessNet, optimizer: torch.optim.Optimizer | None = None, **extra) -> None:
    payload = {"model": model.state_dict(), "model_config": asdict(model.config), **extra}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: str, device: torch.device | str = "cpu") -> tuple[ChessNet, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = ChessNet(ModelConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["model"])
    return model, payload


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
