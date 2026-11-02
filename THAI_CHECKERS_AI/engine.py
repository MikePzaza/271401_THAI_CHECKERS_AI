"""Core Thai checkers (Makhos) rules used by the ADAMS project.

Board encoding
--------------
0  = empty
1  = side +1 man
2  = side +1 king
-1 = side -1 man
-2 = side -1 king

The engine follows the common Thai-checkers profile used by this project:
- 8x8 board, 8 pieces per side on the first two dark-square rows.
- Men move and capture diagonally forward only.
- Captures are compulsory; if the same piece can continue capturing, it must.
- No majority-capture rule: any complete capture route may be chosen.
- Kings are flying kings, but after jumping an enemy they must land on the
  first empty square immediately behind that enemy.
- Captured pieces are removed immediately, so a king may reverse direction
  later in the same capture sequence.
- Promotion during a capture ends that turn immediately.

Tournament organisers can use slightly different draw/clock conventions; draw
handling therefore lives in the server layer instead of this move generator.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

EMPTY = 0
SIDE_ONE = 1
SIDE_TWO = -1
SIDE_ONE_KING = 2
SIDE_TWO_KING = -2
BOARD_SIZE = 8
DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
Board = tuple[tuple[int, ...], ...]
Coord = tuple[int, int]


def inside(r: int, c: int) -> bool:
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


def dark_square(r: int, c: int) -> bool:
    return (r + c) % 2 == 1


def initial_board() -> Board:
    rows = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    for r in (0, 1):
        for c in range(BOARD_SIZE):
            if dark_square(r, c):
                rows[r][c] = SIDE_TWO
    for r in (BOARD_SIZE - 2, BOARD_SIZE - 1):
        for c in range(BOARD_SIZE):
            if dark_square(r, c):
                rows[r][c] = SIDE_ONE
    return tuple(tuple(row) for row in rows)


def validate_board(board: Iterable[Iterable[int]]) -> Board:
    b = tuple(tuple(int(x) for x in row) for row in board)
    if len(b) != BOARD_SIZE or any(len(row) != BOARD_SIZE for row in b):
        raise ValueError(f"board must be {BOARD_SIZE}x{BOARD_SIZE}")
    allowed = {EMPTY, SIDE_ONE, SIDE_TWO, SIDE_ONE_KING, SIDE_TWO_KING}
    if any(cell not in allowed for row in b for cell in row):
        raise ValueError("board contains an invalid piece code")
    return b


def owner(piece: int) -> int:
    return 1 if piece > 0 else -1 if piece < 0 else 0


def belongs(piece: int, side: int) -> bool:
    return piece != EMPTY and owner(piece) == side


def is_king(piece: int) -> bool:
    return abs(piece) == 2


def promotion_row(side: int) -> int:
    return 0 if side == SIDE_ONE else BOARD_SIZE - 1


def promote(piece: int, row: int) -> int:
    if abs(piece) != 1:
        return piece
    side = owner(piece)
    if row == promotion_row(side):
        return 2 * side
    return piece


@dataclass(frozen=True, slots=True)
class Move:
    path: tuple[Coord, ...]
    captures: tuple[Coord, ...] = ()

    @property
    def start(self) -> Coord:
        return self.path[0]

    @property
    def end(self) -> Coord:
        return self.path[-1]

    @property
    def is_capture(self) -> bool:
        return bool(self.captures)

    def to_json(self) -> dict:
        return {
            "path": [list(x) for x in self.path],
            "captures": [list(x) for x in self.captures],
        }


def _board_after_single_jump(
    board: Board,
    start: Coord,
    jumped: Coord,
    end: Coord,
    piece: int,
) -> tuple[Board, int]:
    rows = [list(row) for row in board]
    sr, sc = start
    jr, jc = jumped
    er, ec = end
    rows[sr][sc] = EMPTY
    rows[jr][jc] = EMPTY
    new_piece = promote(piece, er)
    rows[er][ec] = new_piece
    return tuple(tuple(row) for row in rows), new_piece


def _capture_steps(board: Board, r: int, c: int, piece: int) -> list[tuple[Coord, Coord]]:
    """Return possible one-jump (captured-square, landing-square) pairs."""
    steps: list[tuple[Coord, Coord]] = []
    side = owner(piece)

    if not is_king(piece):
        forward = -1 if side == SIDE_ONE else 1
        for dc in (-1, 1):
            mr, mc = r + forward, c + dc
            er, ec = r + 2 * forward, c + 2 * dc
            if not inside(er, ec):
                continue
            if owner(board[mr][mc]) == -side and board[er][ec] == EMPTY:
                steps.append(((mr, mc), (er, ec)))
        return steps

    # Flying king: scan until the first occupied square. It may be captured only
    # when it is an enemy and the square immediately after it is empty.
    for dr, dc in DIRS:
        rr, cc = r + dr, c + dc
        while inside(rr, cc) and board[rr][cc] == EMPTY:
            rr += dr
            cc += dc
        if not inside(rr, cc) or owner(board[rr][cc]) != -side:
            continue
        er, ec = rr + dr, cc + dc
        if inside(er, ec) and board[er][ec] == EMPTY:
            steps.append(((rr, cc), (er, ec)))
    return steps


def _capture_sequences(
    board: Board,
    r: int,
    c: int,
    piece: int,
    path: tuple[Coord, ...],
    captures: tuple[Coord, ...],
) -> list[Move]:
    steps = _capture_steps(board, r, c, piece)
    if not steps:
        return [Move(path, captures)] if captures else []

    result: list[Move] = []
    for jumped, end in steps:
        next_board, next_piece = _board_after_single_jump(board, (r, c), jumped, end, piece)
        next_path = path + (end,)
        next_captures = captures + (jumped,)

        # Thai profile used here: a man that reaches the last rank is crowned,
        # and that capture turn ends immediately.
        if next_piece != piece:
            result.append(Move(next_path, next_captures))
            continue

        continuations = _capture_sequences(
            next_board,
            end[0],
            end[1],
            next_piece,
            next_path,
            next_captures,
        )
        result.extend(continuations or [Move(next_path, next_captures)])
    return result


def _quiet_moves_for_piece(board: Board, r: int, c: int, piece: int) -> list[Move]:
    moves: list[Move] = []
    side = owner(piece)
    if not is_king(piece):
        dr = -1 if side == SIDE_ONE else 1
        for dc in (-1, 1):
            rr, cc = r + dr, c + dc
            if inside(rr, cc) and board[rr][cc] == EMPTY:
                moves.append(Move(((r, c), (rr, cc))))
        return moves

    for dr, dc in DIRS:
        rr, cc = r + dr, c + dc
        while inside(rr, cc) and board[rr][cc] == EMPTY:
            moves.append(Move(((r, c), (rr, cc))))
            rr += dr
            cc += dc
    return moves


@lru_cache(maxsize=300_000)
def legal_moves(board: Board, side: int) -> tuple[Move, ...]:
    if side not in (-1, 1):
        raise ValueError("side must be +1 or -1")

    captures: list[Move] = []
    quiet: list[Move] = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            piece = board[r][c]
            if not belongs(piece, side):
                continue
            captures.extend(_capture_sequences(board, r, c, piece, ((r, c),), ()))
            if not captures:
                quiet.extend(_quiet_moves_for_piece(board, r, c, piece))

    # If a capture exists anywhere, all quiet moves are illegal.
    chosen = captures if captures else quiet
    # Stable order keeps tests and search reproducible.
    return tuple(sorted(set(chosen), key=lambda m: (m.path, m.captures)))


def apply_move(board: Board, move: Move, *, validate: bool = False, side: int | None = None) -> Board:
    if len(move.path) < 2:
        raise ValueError("move path must contain at least start and end")
    if validate:
        if side not in (-1, 1):
            raise ValueError("side is required when validate=True")
        if move not in legal_moves(board, side):
            raise ValueError("illegal move")

    rows = [list(row) for row in board]
    sr, sc = move.start
    er, ec = move.end
    piece = rows[sr][sc]
    if piece == EMPTY:
        raise ValueError("move starts on an empty square")
    rows[sr][sc] = EMPTY
    for cr, cc in move.captures:
        rows[cr][cc] = EMPTY
    rows[er][ec] = promote(piece, er)
    return tuple(tuple(row) for row in rows)


def move_promotes(board: Board, move: Move) -> bool:
    piece = board[move.start[0]][move.start[1]]
    return abs(piece) == 1 and move.end[0] == promotion_row(owner(piece))


def crown_skipped(board: Board, move: Move) -> bool:
    """True ถ้าตานี้เดินถึงแถวสุดท้ายและชนะทันที (ฝ่ายตรงข้ามไม่มีตาเดิน) -> ไม่ต้องให้หุ่นหยิบฮอสมาวาง"""
    if not move_promotes(board, move):
        return False
    piece = board[move.start[0]][move.start[1]]
    return terminal_winner(apply_move(board, move), -owner(piece)) is not None


def terminal_winner(board: Board, side_to_move: int) -> int | None:
    """Return winning side when side_to_move has no legal move, else None."""
    return -side_to_move if not legal_moves(board, side_to_move) else None


def piece_counts(board: Board) -> dict[str, int]:
    flat = [x for row in board for x in row]
    return {
        "side1_men": flat.count(1),
        "side1_kings": flat.count(2),
        "side2_men": flat.count(-1),
        "side2_kings": flat.count(-2),
    }


def board_key(board: Board, side_to_move: int) -> tuple[Board, int]:
    return board, side_to_move
