"""Exact small tablebases for the Thai-checkers rule profile used by this project.

V3 ships an *exact* two-piece (one piece per side) retrograde tablebase.  Unlike
`MicroEndgameDB`, unresolved cycles are classified as draws because the entire
finite two-piece state graph is enumerated before retrograde propagation.

The design deliberately starts at two pieces: it is small enough to build on a
normal laptop and can be independently tested.  Three/four-piece positions keep
using the proof cache until a larger external/offline builder is added.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
from pathlib import Path
import sqlite3
from threading import RLock
from time import perf_counter

from engine import BOARD_SIZE, Board, Move, apply_move, dark_square, legal_moves
from opening_book import board_token, move_from_token, move_token

TABLEBASE_VERSION = 2  # Version 2 is built for the 6x6 board; NOT built for 8x8 (disabled).
DARK_SQUARES = tuple((r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if dark_square(r, c))


@dataclass(slots=True)
class TablebaseResult:
    outcome: int  # +1 win, 0 draw, -1 loss for the side to move
    distance: int
    best_move: Move | None


class ExactTwoPieceTablebase:
    """Persistent exact tablebase for one piece vs one piece.

    Men already sitting on their promotion row are omitted from enumeration,
    since such a position is not reachable through this engine: promotion is
    applied immediately after a move reaches the last rank.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS exact2 (
                key TEXT PRIMARY KEY,
                outcome INTEGER NOT NULL,
                distance INTEGER NOT NULL,
                best_move TEXT,
                version INTEGER NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                name TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    @staticmethod
    def piece_total(board: Board) -> int:
        return sum(1 for row in board for p in row if p)

    @staticmethod
    def _valid_piece_square(piece: int, square: tuple[int, int]) -> bool:
        r, _ = square
        if piece == 1 and r == 0:
            return False
        if piece == -1 and r == BOARD_SIZE - 1:
            return False
        return True

    @staticmethod
    def _make_board(plus_square: tuple[int, int], minus_square: tuple[int, int], plus_piece: int, minus_piece: int) -> Board:
        rows = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        rows[plus_square[0]][plus_square[1]] = plus_piece
        rows[minus_square[0]][minus_square[1]] = minus_piece
        return tuple(tuple(r) for r in rows)

    @classmethod
    def enumerate_states(cls) -> list[tuple[Board, int]]:
        states: list[tuple[Board, int]] = []
        for ps in DARK_SQUARES:
            for ms in DARK_SQUARES:
                if ps == ms:
                    continue
                for pp in (1, 2):
                    if not cls._valid_piece_square(pp, ps):
                        continue
                    for mp in (-1, -2):
                        if not cls._valid_piece_square(mp, ms):
                            continue
                        board = cls._make_board(ps, ms, pp, mp)
                        states.append((board, 1))
                        states.append((board, -1))
        return states

    def is_built(self) -> bool:
        row = self.conn.execute("SELECT value FROM metadata WHERE name='exact2_complete'").fetchone()
        return bool(row and row[0] == str(TABLEBASE_VERSION))

    def lookup(self, board: Board, side: int) -> TablebaseResult | None:
        if self.piece_total(board) != 2:
            return None
        with self._lock:
            row = self.conn.execute(
                "SELECT outcome,distance,best_move FROM exact2 WHERE key=? AND version=?",
                (board_token(board, side), TABLEBASE_VERSION),
            ).fetchone()
        if row is None:
            return None
        move = move_from_token(row[2], board, side) if row[2] else None
        return TablebaseResult(int(row[0]), int(row[1]), move)

    def build(self, *, force: bool = False) -> dict:
        if self.is_built() and not force:
            return self.stats() | {"rebuilt": False}

        started = perf_counter()
        states = self.enumerate_states()
        index = {state: i for i, state in enumerate(states)}
        n = len(states)
        # None = unresolved; exact draws are assigned after retrograde reaches a fixpoint.
        outcome: list[int | None] = [None] * n
        distance = [0] * n
        best: list[Move | None] = [None] * n
        remaining = [0] * n
        loss_delay = [-1] * n
        loss_delay_move: list[Move | None] = [None] * n
        predecessors: list[list[tuple[int, Move]]] = [[] for _ in range(n)]
        heap: list[tuple[int, int]] = []
        edges = 0

        for i, (board, side) in enumerate(states):
            moves = legal_moves(board, side)
            if not moves:
                outcome[i] = -1
                distance[i] = 0
                heapq.heappush(heap, (0, i))
                continue

            two_piece_successors = 0
            immediate_win: Move | None = None
            for move in moves:
                child = apply_move(board, move)
                if self.piece_total(child) < 2:
                    immediate_win = move
                    break
                child_state = (child, -side)
                child_id = index.get(child_state)
                if child_id is None:
                    # This should only happen for invalid/unreachable hand-built
                    # positions. Enumeration covers every reachable 1v1 state.
                    continue
                predecessors[child_id].append((i, move))
                two_piece_successors += 1
                edges += 1

            if immediate_win is not None:
                outcome[i] = 1
                distance[i] = 1
                best[i] = immediate_win
                heapq.heappush(heap, (1, i))
            else:
                remaining[i] = two_piece_successors
                # Defensive fallback: if no modeled successor exists, legal play
                # cannot continue in the enumerated graph, so treat as loss.
                if two_piece_successors == 0:
                    outcome[i] = -1
                    heapq.heappush(heap, (0, i))

        # Process solved nodes in increasing distance. This makes the first loss
        # child that proves a predecessor win also its shortest known forced win.
        while heap:
            _, child_id = heapq.heappop(heap)
            child_outcome = outcome[child_id]
            child_distance = distance[child_id]
            if child_outcome is None:
                continue
            for pred_id, pred_move in predecessors[child_id]:
                if outcome[pred_id] is not None:
                    continue
                if child_outcome == -1:
                    outcome[pred_id] = 1
                    distance[pred_id] = child_distance + 1
                    best[pred_id] = pred_move
                    heapq.heappush(heap, (distance[pred_id], pred_id))
                elif child_outcome == 1:
                    remaining[pred_id] -= 1
                    if child_distance > loss_delay[pred_id]:
                        loss_delay[pred_id] = child_distance
                        loss_delay_move[pred_id] = pred_move
                    if remaining[pred_id] == 0:
                        outcome[pred_id] = -1
                        distance[pred_id] = max(0, loss_delay[pred_id]) + 1
                        best[pred_id] = loss_delay_move[pred_id]
                        heapq.heappush(heap, (distance[pred_id], pred_id))

        # Anything left unresolved belongs to the draw region. Choose a move that
        # stays inside the draw region when possible.
        unresolved = {i for i, value in enumerate(outcome) if value is None}
        draws = len(unresolved)
        for i in unresolved:
            outcome[i] = 0
        for i in unresolved:
            board, side = states[i]
            for move in legal_moves(board, side):
                child = apply_move(board, move)
                if self.piece_total(child) != 2:
                    continue
                child_id = index.get((child, -side))
                if child_id in unresolved:
                    best[i] = move
                    break
            if best[i] is None:
                moves = legal_moves(board, side)
                best[i] = moves[0] if moves else None

        rows = []
        wins = losses = 0
        for i, (board, side) in enumerate(states):
            out = int(outcome[i])
            if out > 0:
                wins += 1
            elif out < 0:
                losses += 1
            token = move_token(best[i]) if best[i] else None
            rows.append((board_token(board, side), out, int(distance[i]), token, TABLEBASE_VERSION))

        with self._lock:
            self.conn.execute("DELETE FROM exact2")
            self.conn.executemany(
                "INSERT OR REPLACE INTO exact2(key,outcome,distance,best_move,version) VALUES (?,?,?,?,?)",
                rows,
            )
            self.conn.execute(
                "INSERT OR REPLACE INTO metadata(name,value) VALUES ('exact2_complete',?)",
                (str(TABLEBASE_VERSION),),
            )
            self.conn.commit()

        return {
            "states": n,
            "edges": edges,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "elapsed_s": round(perf_counter() - started, 3),
            "rebuilt": True,
            "version": TABLEBASE_VERSION,
        }

    def stats(self) -> dict:
        with self._lock:
            count = int(self.conn.execute("SELECT COUNT(*) FROM exact2 WHERE version=?", (TABLEBASE_VERSION,)).fetchone()[0])
            rows = self.conn.execute(
                "SELECT outcome,COUNT(*) FROM exact2 WHERE version=? GROUP BY outcome",
                (TABLEBASE_VERSION,),
            ).fetchall()
        by_outcome = {int(k): int(v) for k, v in rows}
        return {
            "built": self.is_built(),
            "positions": count,
            "wins": by_outcome.get(1, 0),
            "draws": by_outcome.get(0, 0),
            "losses": by_outcome.get(-1, 0),
            "version": TABLEBASE_VERSION,
        }

    def close(self) -> None:
        with self._lock:
            self.conn.close()
