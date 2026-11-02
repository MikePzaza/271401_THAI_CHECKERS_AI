"""Offline 8x8 human-vs-ADAMS game state and 1..64 notation.

Coordinates are from the human's viewpoint: bottom left is #1, top right
is #64. A numbered light square exists visually but cannot hold a piece.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re

from engine import (
    BOARD_SIZE, Board, Move, SIDE_ONE, SIDE_TWO, apply_move, board_key,
    initial_board, legal_moves, move_promotes, terminal_winner,
)

HUMAN = SIDE_ONE
AI = SIDE_TWO
TOTAL_SQUARES = BOARD_SIZE * BOARD_SIZE


def number_of(coord: tuple[int, int]) -> int:
    r, c = coord
    if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
        raise ValueError('พิกัดอยู่นอกกระดาน')
    return (BOARD_SIZE - 1 - r) * BOARD_SIZE + c + 1


def coord_of(number: int) -> tuple[int, int]:
    if not 1 <= number <= TOTAL_SQUARES:
        raise ValueError('กรุณาใส่เลขช่องตั้งแต่ 1 ถึง 64')
    return (BOARD_SIZE - 1 - (number - 1) // BOARD_SIZE, (number - 1) % BOARD_SIZE)


def move_numbers(move: Move) -> tuple[int, ...]:
    return tuple(number_of(coord) for coord in move.path)


def move_label(move: Move) -> str:
    return ' → '.join(map(str, move_numbers(move)))


def parse_numbers(raw: str) -> tuple[int, ...]:
    """Accept '8 13', '8-13', '8,13', or full multi-jump paths."""
    if not raw.strip():
        raise ValueError('กรุณาใส่เลขอย่างน้อย 2 ช่อง เช่น 8 13')
    if not re.fullmatch(r'[0-9\s,;>\-→]+', raw):
        raise ValueError('กรุณาใช้เฉพาะเลขช่อง เช่น 8 13 (ไม่ต้องใส่ชื่อหมาก)')
    parts = [x for x in re.split(r'[\s,;>\-→]+', raw.strip()) if x]
    if len(parts) < 2:
        raise ValueError('กรุณาใส่เลขต้นทางและปลายทาง เช่น 8 13')
    return tuple(number_of(coord_of(int(x))) for x in parts)


def resolve_human_move(board: Board, raw: str) -> Move:
    path = parse_numbers(raw)
    moves = legal_moves(board, HUMAN)
    if not moves:
        raise ValueError('ไม่มีตาเดินที่ถูกกติกาสำหรับคน')

    # Exact route, especially relevant when a capture takes several jumps.
    matches = [m for m in moves if move_numbers(m) == path]
    if not matches and len(path) == 2:
        # A start + final destination suffices for a unique capture route.
        matches = [m for m in moves if move_numbers(m)[0] == path[0]
                   and move_numbers(m)[-1] == path[-1]]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        options = ' | '.join(move_label(m) for m in matches[:5])
        raise ValueError(f'มีหลายเส้นทางที่เริ่ม/จบเหมือนกัน กรุณาใส่เลขทุกจุดที่กระโดด เช่น {options}')

    from_here = [move_label(m) for m in moves if move_numbers(m)[0] == path[0]]
    if from_here:
        raise ValueError('เดินแบบนี้ไม่ได้ จากช่องนี้เดินได้: ' + ', '.join(from_here[:8]))
    if any(m.is_capture for m in moves):
        raise ValueError('ต้องกินหมากก่อน! ตาที่เดินได้: ' + ', '.join(move_label(m) for m in moves[:6]))
    raise ValueError('ช่องต้นทางนี้เดินไม่ได้ กรุณาดูรายการ “ตาที่คนเดินได้”')


@dataclass(frozen=True)
class HistoryEntry:
    ply: int
    side: int
    move: Move
    captures: tuple[int, ...]
    promoted: bool

    @property
    def label(self) -> str:
        who = 'คน' if self.side == HUMAN else 'ADAMS'
        suffix = ''
        if self.captures:
            suffix += '  [กิน ' + ', '.join(map(str, self.captures)) + ']'
        if self.promoted:
            suffix += '  [เป็นฮอส ★]'
        return f'{self.ply:02d}. {who:<5} {move_label(self.move)}{suffix}'


class CheckersGame:
    def __init__(self) -> None:
        self.board: Board = initial_board()
        self.side_to_move = HUMAN
        self.history: list[HistoryEntry] = []
        self.status = 'playing'  # playing, win, draw
        self.winner: int | None = None
        self.draw_reason: str | None = None
        self.last_move: Move | None = None
        self.repetitions: Counter = Counter({board_key(self.board, HUMAN): 1})
        self.quiet_plies = 0

    @property
    def human_moves(self) -> tuple[Move, ...]:
        if self.status != 'playing' or self.side_to_move != HUMAN:
            return ()
        return legal_moves(self.board, HUMAN)

    def play_human(self, raw: str) -> HistoryEntry:
        if self.status != 'playing' or self.side_to_move != HUMAN:
            raise ValueError('ตอนนี้ยังไม่ถึงตาของคุณ')
        move = resolve_human_move(self.board, raw)
        return self._move(move, HUMAN)

    def play_ai(self, move: Move) -> HistoryEntry:
        if self.status != 'playing' or self.side_to_move != AI:
            raise ValueError('ตอนนี้ยังไม่ถึงตาของ AI')
        if move not in legal_moves(self.board, AI):
            raise ValueError('AI ส่งตาเดินที่ผิดกติกา')
        return self._move(move, AI)

    def _move(self, move: Move, side: int) -> HistoryEntry:
        before = self.board
        promoted = move_promotes(before, move)
        record = HistoryEntry(len(self.history) + 1, side, move,
                              tuple(number_of(c) for c in move.captures), promoted)
        self.board = apply_move(before, move)
        self.side_to_move = -side
        self.history.append(record)
        self.last_move = move
        self.quiet_plies = 0 if move.captures or promoted else self.quiet_plies + 1

        # Exactly one king per side ends the game immediately after this move.
        if sorted(p for row in self.board for p in row if p) == [-2, 2]:
            self.status, self.winner = 'draw', None
            self.draw_reason = 'เหลือฮอสฝ่ายละ 1 ตัว'
            return record

        winner = terminal_winner(self.board, self.side_to_move)
        if winner is not None:
            self.status, self.winner = 'win', winner
        else:
            key = board_key(self.board, self.side_to_move)
            self.repetitions[key] += 1
            if self.repetitions[key] >= 3:
                self.status, self.draw_reason = 'draw', 'ตำแหน่งเดิมเกิดซ้ำ 3 ครั้ง'
            elif self.quiet_plies >= 80:
                self.status, self.draw_reason = 'draw', 'ครบ 80 ครึ่งตาโดยไม่มีการกิน/เลื่อนเป็นฮอส'
        return record
