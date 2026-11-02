"""Persistent opening book learned from self-play.

The book is intentionally data-driven: no hard-coded claim is made that a move
is theoretically optimal.  Entries store empirical W/D/L statistics from the
side-to-move's perspective and are only used after a minimum number of visits.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from math import log, sqrt
from pathlib import Path
from threading import RLock

from engine import Board, Move, legal_moves


def board_token(board: Board, side: int) -> str:
    flat = ",".join(str(x) for row in board for x in row)
    return f"{side}|{flat}"


def move_token(move: Move) -> str:
    return ";".join(f"{r},{c}" for r, c in move.path)


def move_from_token(token: str, board: Board, side: int) -> Move | None:
    try:
        path = tuple(tuple(int(v) for v in item.split(",")) for item in token.split(";"))
    except (TypeError, ValueError):
        return None
    for move in legal_moves(board, side):
        if move.path == path:
            return move
    return None


@dataclass(slots=True)
class BookChoice:
    move: Move
    visits: int
    score: float
    wins: int
    draws: int
    losses: int


class OpeningBook:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._lock = RLock()
        self.positions: dict[str, dict[str, dict[str, int]]] = {}
        if self.path and self.path.exists():
            self.load()

    def load(self) -> None:
        if not self.path or not self.path.exists():
            return
        with self._lock:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.positions = raw.get("positions", {}) if isinstance(raw, dict) else {}

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "positions": self.positions}
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def record(self, board: Board, side: int, move: Move, result_for_side: int) -> None:
        """Record -1/0/+1 final result from the side-to-move's perspective."""
        if result_for_side not in (-1, 0, 1):
            raise ValueError("result_for_side must be -1, 0 or +1")
        pkey = board_token(board, side)
        mkey = move_token(move)
        with self._lock:
            entry = self.positions.setdefault(pkey, {}).setdefault(
                mkey, {"visits": 0, "wins": 0, "draws": 0, "losses": 0}
            )
            entry["visits"] += 1
            if result_for_side > 0:
                entry["wins"] += 1
            elif result_for_side < 0:
                entry["losses"] += 1
            else:
                entry["draws"] += 1

    @staticmethod
    def _lower_confidence(wins: int, draws: int, visits: int) -> float:
        """Wilson-like conservative score using draw as half a win."""
        if visits <= 0:
            return -1.0
        p = (wins + 0.5 * draws) / visits
        z = 1.281551565545  # ~80% interval; conservative but not too sparse.
        den = 1 + z * z / visits
        centre = p + z * z / (2 * visits)
        margin = z * sqrt((p * (1 - p) + z * z / (4 * visits)) / visits)
        return (centre - margin) / den

    def choose(self, board: Board, side: int, *, min_visits: int = 3) -> BookChoice | None:
        pkey = board_token(board, side)
        with self._lock:
            entries = dict(self.positions.get(pkey, {}))
        legal = {move_token(m): m for m in legal_moves(board, side)}
        choices: list[BookChoice] = []
        for token, stats in entries.items():
            move = legal.get(token)
            if move is None:
                continue
            visits = int(stats.get("visits", 0))
            if visits < min_visits:
                continue
            wins = int(stats.get("wins", 0))
            draws = int(stats.get("draws", 0))
            losses = int(stats.get("losses", 0))
            conservative = self._lower_confidence(wins, draws, visits)
            # Tiny visit bonus breaks near-ties in favour of better-tested moves.
            score = conservative + 0.002 * log(visits + 1)
            choices.append(BookChoice(move, visits, score, wins, draws, losses))
        if not choices:
            return None
        return max(choices, key=lambda x: (x.score, x.visits, move_token(x.move)))

    def summary(self) -> dict[str, int]:
        with self._lock:
            positions = len(self.positions)
            moves = sum(len(v) for v in self.positions.values())
            visits = sum(
                int(stats.get("visits", 0))
                for entries in self.positions.values()
                for stats in entries.values()
            )
        return {"positions": positions, "moves": moves, "visits": visits}
