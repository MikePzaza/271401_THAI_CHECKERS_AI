"""Persistent micro endgame database for Thai checkers.

This is a *proof cache*, not a complete solved-game tablebase.  It searches very
small positions and persists only outcomes that can be proven as a win/loss
within the configured proof horizon. Unknown cyclic/deep branches are left
unknown rather than mislabeled as draws. This makes it safe to use as a tactical
oracle while the project grows toward a full retrograde tablebase.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from threading import RLock
from time import monotonic

from engine import Board, Move, apply_move, legal_moves
from opening_book import board_token, move_from_token, move_token


@dataclass(slots=True)
class EndgameResult:
    outcome: int  # +1 = side to move can force win, -1 = forced loss
    distance: int
    best_move: Move | None
    nodes: int = 0


class MicroEndgameDB:
    def __init__(
        self,
        path: str | Path,
        *,
        max_pieces: int = 4,
        proof_depth: int = 24,
        node_limit: int = 120_000,
    ):
        self.path = Path(path)
        self.max_pieces = max_pieces
        self.proof_depth = proof_depth
        self.node_limit = node_limit
        self._lock = RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS positions (
                key TEXT PRIMARY KEY,
                outcome INTEGER NOT NULL,
                distance INTEGER NOT NULL,
                best_move TEXT,
                proof_depth INTEGER NOT NULL
            )
            """
        )
        self.conn.commit()

    @staticmethod
    def piece_total(board: Board) -> int:
        return sum(1 for row in board for p in row if p)

    def _key(self, board: Board, side: int) -> str:
        return board_token(board, side)

    def lookup(self, board: Board, side: int) -> EndgameResult | None:
        if self.piece_total(board) > self.max_pieces:
            return None
        key = self._key(board, side)
        with self._lock:
            row = self.conn.execute(
                "SELECT outcome, distance, best_move FROM positions WHERE key=?", (key,)
            ).fetchone()
        if not row:
            return None
        move = move_from_token(row[2], board, side) if row[2] else None
        return EndgameResult(int(row[0]), int(row[1]), move, 0)

    def _store(self, board: Board, side: int, result: EndgameResult) -> None:
        token = move_token(result.best_move) if result.best_move else None
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO positions(key,outcome,distance,best_move,proof_depth) VALUES (?,?,?,?,?)",
                (self._key(board, side), result.outcome, result.distance, token, self.proof_depth),
            )
            self.conn.commit()

    def solve(self, board: Board, side: int, *, max_time_ms: int | None = None) -> EndgameResult | None:
        cached = self.lookup(board, side)
        if cached:
            return cached
        if self.piece_total(board) > self.max_pieces:
            return None

        memo: dict[tuple[Board, int, int], tuple[int, int, Move | None] | None] = {}
        active: set[tuple[Board, int]] = set()
        nodes = 0
        deadline = monotonic() + max_time_ms / 1000 if max_time_ms is not None else None

        def dfs(b: Board, s: int, depth_left: int) -> tuple[int, int, Move | None] | None:
            nonlocal nodes
            nodes += 1
            if nodes > self.node_limit or depth_left < 0:
                return None
            if deadline is not None and monotonic() >= deadline:
                return None
            pos = (b, s)
            if pos in active:
                # A cycle is not enough to prove a draw without repetition count.
                return None
            key = (b, s, depth_left)
            if key in memo:
                return memo[key]

            # Avoid a SQLite round-trip at every recursive node. The root was
            # checked before search; recursive proofs use the local memo table.
            moves = legal_moves(b, s)
            if not moves:
                ans = (-1, 0, None)
                memo[key] = ans
                return ans
            if depth_left == 0:
                memo[key] = None
                return None

            active.add(pos)
            child_results: list[tuple[Move, tuple[int, int, Move | None] | None]] = []
            try:
                for move in moves:
                    child = dfs(apply_move(b, move), -s, depth_left - 1)
                    child_results.append((move, child))
                    if child is not None and child[0] == -1:
                        # Opponent is proven lost. Prefer the shortest win.
                        wins = [
                            (m, r) for m, r in child_results
                            if r is not None and r[0] == -1
                        ]
                        move_best, result_best = min(wins, key=lambda x: x[1][1])
                        ans = (1, result_best[1] + 1, move_best)
                        memo[key] = ans
                        return ans
            finally:
                active.discard(pos)

            # Current side is proven lost only when every legal move is proven
            # to give the opponent a win. Any unknown branch keeps us unknown.
            if child_results and all(r is not None and r[0] == 1 for _, r in child_results):
                move_best, result_best = max(child_results, key=lambda x: x[1][1])  # delay mate
                ans = (-1, result_best[1] + 1, move_best)
                memo[key] = ans
                return ans
            memo[key] = None
            return None

        raw = dfs(board, side, self.proof_depth)
        if raw is None:
            return None
        result = EndgameResult(raw[0], raw[1], raw[2], nodes)
        self._store(board, side, result)
        return result

    def stats(self) -> dict[str, int]:
        with self._lock:
            count = self.conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
        return {
            "cached_positions": int(count),
            "max_pieces": self.max_pieces,
            "proof_depth": self.proof_depth,
        }

    def close(self) -> None:
        with self._lock:
            self.conn.close()
