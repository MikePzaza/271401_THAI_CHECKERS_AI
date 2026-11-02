"""ADAMS V3 — Adaptive Deep Analysis and Move Selection.

V3 keeps the explainable alpha-beta core and adds project-specific composition:

* danger/volatility root profiling;
* confidence-guided root search allocation (CGRA);
* aspiration re-search whose window scales with instability/volatility;
* principal-variation search (PVS) inside the tree;
* tactical quiescence + bounded forced-line extensions;
* stability-aware final move selection;
* transposition, killer/history ordering and late quiet-move reductions;
* empirical opening book;
* exact two-piece retrograde tablebase + 3/4-piece proof cache;
* loadable evaluation weights trained by the V3 population self-play trainer.

ADAMS V3 is a research architecture developed for this project.  Individual
search techniques such as alpha-beta, PVS and aspiration windows are established
ideas; novelty must be argued and measured at the composition/ablation level.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from math import inf
from pathlib import Path
from statistics import mean
from time import monotonic

from engine import BOARD_SIZE, Board, Move, apply_move, is_king, legal_moves, move_promotes, owner, promotion_row
from opening_book import OpeningBook
from endgame_db import MicroEndgameDB
from tablebase import ExactTwoPieceTablebase

MATE = 1_000_000
MODEL_VERSION = "ADAMS-V3.0"


class SearchTimeout(RuntimeError):
    pass


@dataclass(slots=True)
class EvaluationWeights:
    man_value: float = 100.0
    king_opening: float = 220.0
    king_midgame: float = 245.0
    king_endgame: float = 285.0
    advance_opening: float = 7.0
    advance_midgame: float = 5.0
    advance_endgame: float = 3.0
    center_man: float = 3.0
    center_king: float = 1.0
    back_guard: float = 5.0
    promotion_pressure: float = 18.0
    mobility_opening: float = 2.0
    mobility_midgame: float = 3.5
    mobility_endgame: float = 5.0
    trapped_penalty: float = 8.0
    formation: float = 2.5
    capture_pressure_opening: float = 1.5
    capture_pressure_midgame: float = 2.5
    capture_pressure_endgame: float = 4.0

    def to_json(self) -> dict[str, float]:
        return {k: round(float(v), 6) for k, v in asdict(self).items()}

    @classmethod
    def from_file(cls, path: str | Path) -> "EvaluationWeights":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if "weights" in raw:
            raw = raw["weights"]
        allowed = set(cls.__dataclass_fields__)
        values = {k: float(v) for k, v in raw.items() if k in allowed}
        return cls(**values)

    def save(self, path: str | Path, *, metadata: dict | None = None) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model_version": MODEL_VERSION, "weights": self.to_json()}
        if metadata:
            payload["metadata"] = metadata
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


DEFAULT_WEIGHTS = EvaluationWeights()


@dataclass(slots=True)
class SearchConfig:
    max_depth: int = 7
    time_limit_ms: int = 700
    quiescence_depth: int = 5
    extension_budget: int = 2
    stability_penalty: float = 0.08
    tt_max_entries: int = 300_000
    book_min_visits: int = 12
    use_opening_book: bool = True
    use_tablebase: bool = True
    use_endgame_db: bool = True
    use_pvs: bool = True
    use_aspiration: bool = True
    use_cgra: bool = True
    aspiration_base: float = 42.0
    focus_after_depth: int = 3
    focus_candidates: int = 3
    confidence_margin: float = 38.0
    persistent_tt: bool = True


@dataclass(slots=True)
class CandidateProfile:
    move: Move
    static_score: float
    volatility: float
    danger: float
    extra_depth: int
    latest_score: float | None = None

    def to_json(self) -> dict:
        return {
            "move": self.move.to_json(),
            "static_score": round(self.static_score, 2),
            "volatility": round(self.volatility, 2),
            "danger": round(self.danger, 2),
            "extra_depth": self.extra_depth,
            "latest_score": None if self.latest_score is None else round(self.latest_score, 2),
        }


@dataclass(slots=True)
class SearchResult:
    move: Move | None
    score: float
    depth: int
    nodes: int
    elapsed_ms: float
    principal_variation: list[Move] = field(default_factory=list)
    candidates: list[CandidateProfile] = field(default_factory=list)
    source: str = "search"
    model_version: str = MODEL_VERSION
    extra: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "move": None if self.move is None else self.move.to_json(),
            "score": round(self.score, 2),
            "depth": self.depth,
            "nodes": self.nodes,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "principal_variation": [m.to_json() for m in self.principal_variation],
            "candidates": [c.to_json() for c in self.candidates[:6]],
            "source": self.source,
            "model_version": self.model_version,
            "extra": self.extra,
        }


@dataclass(slots=True)
class TTEntry:
    depth: int
    value: float
    flag: str
    best_move: Move | None


def _phase(board: Board) -> str:
    total = sum(1 for row in board for p in row if p)
    if total >= 10:
        return "opening"
    if total >= 5:
        return "midgame"
    return "endgame"


def _side_features(board: Board, side: int, weights: EvaluationWeights) -> dict[str, float]:
    phase = _phase(board)
    king_value = {
        "opening": weights.king_opening,
        "midgame": weights.king_midgame,
        "endgame": weights.king_endgame,
    }[phase]
    advance_weight = {
        "opening": weights.advance_opening,
        "midgame": weights.advance_midgame,
        "endgame": weights.advance_endgame,
    }[phase]

    material = advancement = center = back_guard = promotion_pressure = 0.0
    piece_mobility_hint = 0.0
    formation = 0.0

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p = board[r][c]
            if owner(p) != side:
                continue
            if is_king(p):
                material += king_value
                center += weights.center_king * (
                    max(0.0, (BOARD_SIZE - 1) / 2 - abs((BOARD_SIZE - 1) / 2 - r)) + max(0.0, (BOARD_SIZE - 1) / 2 - abs((BOARD_SIZE - 1) / 2 - c))
                )
            else:
                material += weights.man_value
                progress = (BOARD_SIZE - 1 - r) if side == 1 else r
                advancement += progress * advance_weight
                if r == (BOARD_SIZE - 1 if side == 1 else 0):
                    back_guard += weights.back_guard
                if abs(r - promotion_row(side)) == 1:
                    promotion_pressure += weights.promotion_pressure
                if 1 < r < BOARD_SIZE - 2 and 1 < c < BOARD_SIZE - 2:
                    center += weights.center_man

                # Cheap local freedom hint. Full legal mobility is added below.
                dr = -1 if side == 1 else 1
                free = 0
                for dc in (-1, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and board[rr][cc] == 0:
                        free += 1
                if free == 0:
                    piece_mobility_hint -= weights.trapped_penalty

    # Local diagonal cohesion: each friendly adjacent pair contributes once.
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if owner(board[r][c]) != side:
                continue
            for dr, dc in ((1, 1), (1, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and owner(board[rr][cc]) == side:
                    formation += weights.formation

    legal = legal_moves(board, side)
    mobility = len(legal)
    capture_pressure = float(sum(len(m.captures) for m in legal if m.captures))
    return {
        "material": material,
        "advancement": advancement,
        "center": center,
        "back_guard": back_guard,
        "promotion_pressure": promotion_pressure,
        "mobility": float(mobility),
        "freedom": piece_mobility_hint,
        "formation": formation,
        "capture_pressure": capture_pressure,
    }


def evaluate(board: Board, perspective: int, weights: EvaluationWeights | None = None) -> float:
    """Dynamic handcrafted evaluation. Positive is good for perspective."""
    w = weights or DEFAULT_WEIGHTS
    mine = _side_features(board, perspective, w)
    theirs = _side_features(board, -perspective, w)
    phase = _phase(board)
    mobility_weight = {
        "opening": w.mobility_opening,
        "midgame": w.mobility_midgame,
        "endgame": w.mobility_endgame,
    }[phase]
    pressure_weight = {
        "opening": w.capture_pressure_opening,
        "midgame": w.capture_pressure_midgame,
        "endgame": w.capture_pressure_endgame,
    }[phase]

    score = 0.0
    for key in ("material", "advancement", "center", "back_guard", "promotion_pressure", "freedom", "formation"):
        score += mine[key] - theirs[key]
    score += mobility_weight * (mine["mobility"] - theirs["mobility"])
    score += pressure_weight * (mine["capture_pressure"] - theirs["capture_pressure"])
    return score


class ADAMS:
    def __init__(
        self,
        config: SearchConfig | None = None,
        *,
        weights: EvaluationWeights | None = None,
        opening_book: OpeningBook | None = None,
        tablebase: ExactTwoPieceTablebase | None = None,
        endgame_db: MicroEndgameDB | None = None,
    ):
        self.config = config or SearchConfig()
        self.weights = weights or DEFAULT_WEIGHTS
        self.opening_book = opening_book
        self.tablebase = tablebase
        self.endgame_db = endgame_db
        self.nodes = 0
        self.deadline = inf
        self.root_side = -1
        self.tt: dict[tuple[Board, int, int], TTEntry] = {}
        self.history: dict[tuple[int, Move], int] = {}
        self.killers: dict[int, list[Move]] = {}

    def _eval(self, board: Board, side: int | None = None) -> float:
        return evaluate(board, self.root_side if side is None else side, self.weights)

    def _check_time(self) -> None:
        if (self.nodes & 7) == 0 and monotonic() >= self.deadline:
            raise SearchTimeout

    def _capture_severity(self, moves: tuple[Move, ...]) -> float:
        captures = [len(m.captures) for m in moves if m.captures]
        if not captures:
            return 0.0
        return 28.0 * max(captures) + 8.0 * len(captures)

    def _profile_candidates(self, board: Board, side: int, moves: tuple[Move, ...]) -> list[CandidateProfile]:
        baseline = evaluate(board, side, self.weights)
        profiles: list[CandidateProfile] = []
        for move in moves:
            child = apply_move(board, move)
            static = evaluate(child, side, self.weights)
            replies = legal_moves(child, -side)
            danger = self._capture_severity(replies)
            tactical_gain = 32.0 * len(move.captures)
            promo = 42.0 if move_promotes(board, move) else 0.0
            swing = abs(static - baseline)
            volatility = swing + danger + tactical_gain + promo + min(len(replies), 12) * 1.5
            extra = 0
            if volatility >= 55:
                extra += 1
            if volatility >= 125 or len(move.captures) >= 2 or move_promotes(board, move):
                extra += 1
            profiles.append(CandidateProfile(move, static, volatility, danger, min(extra, 2)))
        profiles.sort(key=lambda p: p.static_score + 0.20 * p.volatility - 0.12 * p.danger, reverse=True)
        return profiles

    def _move_order_score(self, board: Board, side: int, move: Move, ply: int, tt_move: Move | None) -> float:
        if move == tt_move:
            return 10_000_000
        score = 0.0
        if move.captures:
            score += 1_000_000 + 20_000 * len(move.captures)
        if move_promotes(board, move):
            score += 120_000
        if move in self.killers.get(ply, ()):
            score += 50_000
        score += self.history.get((side, move), 0)
        return score

    def _ordered_moves(self, board: Board, side: int, moves: tuple[Move, ...], ply: int, tt_move: Move | None) -> list[Move]:
        return sorted(moves, key=lambda m: self._move_order_score(board, side, m, ply, tt_move), reverse=True)

    def _remember_killer(self, ply: int, move: Move) -> None:
        if move.captures:
            return
        arr = self.killers.setdefault(ply, [])
        if move in arr:
            arr.remove(move)
        arr.insert(0, move)
        del arr[2:]

    def _tablebase_value(self, board: Board, side: int, ply: int) -> float | None:
        if not self.config.use_tablebase or self.tablebase is None:
            return None
        result = self.tablebase.lookup(board, side)
        if result is None:
            return None
        if result.outcome == 0:
            return 0.0
        sign_from_root = result.outcome if side == self.root_side else -result.outcome
        return sign_from_root * (MATE - min(10_000, ply + result.distance))

    def _endgame_value(self, board: Board, side: int, ply: int) -> float | None:
        if not self.config.use_endgame_db or self.endgame_db is None:
            return None
        result = self.endgame_db.lookup(board, side)
        if result is None:
            return None
        sign_from_root = result.outcome if side == self.root_side else -result.outcome
        return sign_from_root * (MATE - min(10_000, ply + result.distance))

    def _quiescence(self, board: Board, side: int, alpha: float, beta: float, ply: int, depth_left: int) -> float:
        self.nodes += 1
        self._check_time()
        moves = legal_moves(board, side)
        if not moves:
            return (-MATE + ply) if side == self.root_side else (MATE - ply)
        exact = self._tablebase_value(board, side, ply)
        if exact is not None:
            return exact
        endgame = self._endgame_value(board, side, ply)
        if endgame is not None:
            return endgame
        stand_pat = self._eval(board)
        capture_moves = tuple(m for m in moves if m.captures)
        if depth_left <= 0 or not capture_moves:
            return stand_pat

        ordered = sorted(capture_moves, key=lambda m: len(m.captures), reverse=True)
        if side == self.root_side:
            value = stand_pat
            if value >= beta:
                return value
            alpha = max(alpha, value)
            for move in ordered:
                value = max(value, self._quiescence(apply_move(board, move), -side, alpha, beta, ply + 1, depth_left - 1))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        value = stand_pat
        if value <= alpha:
            return value
        beta = min(beta, value)
        for move in ordered:
            value = min(value, self._quiescence(apply_move(board, move), -side, alpha, beta, ply + 1, depth_left - 1))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value

    def _search(self, board: Board, side: int, depth: int, alpha: float, beta: float, ply: int, extension_left: int) -> float:
        self.nodes += 1
        self._check_time()
        alpha0, beta0 = alpha, beta

        exact = self._tablebase_value(board, side, ply)
        if exact is not None:
            return exact
        endgame = self._endgame_value(board, side, ply)
        if endgame is not None:
            return endgame

        key = (board, side, self.root_side)
        entry = self.tt.get(key)
        if entry and entry.depth >= depth:
            if entry.flag == "EXACT":
                return entry.value
            if entry.flag == "LOWER":
                alpha = max(alpha, entry.value)
            elif entry.flag == "UPPER":
                beta = min(beta, entry.value)
            if alpha >= beta:
                return entry.value

        moves = legal_moves(board, side)
        if not moves:
            return (-MATE + ply) if side == self.root_side else (MATE - ply)
        if depth <= 0:
            return self._quiescence(board, side, alpha, beta, ply, self.config.quiescence_depth)

        local_depth = depth
        next_extension = extension_left
        if extension_left > 0 and moves[0].captures and (len(moves) == 1 or max(len(m.captures) for m in moves) >= 2):
            local_depth += 1
            next_extension -= 1

        tt_move = entry.best_move if entry else None
        ordered = self._ordered_moves(board, side, moves, ply, tt_move)
        best_move: Move | None = None

        if side == self.root_side:
            value = -inf
            for idx, move in enumerate(ordered):
                child = apply_move(board, move)
                child_depth = local_depth - 1
                reduced = idx >= 4 and child_depth >= 2 and not move.captures and not move_promotes(board, move)
                probe_depth = child_depth - 1 if reduced else child_depth
                if self.config.use_pvs and idx > 0 and alpha + 1.0 < beta:
                    score = self._search(child, -side, probe_depth, alpha, alpha + 1.0, ply + 1, next_extension)
                    if alpha < score < beta:
                        score = self._search(child, -side, probe_depth, alpha, beta, ply + 1, next_extension)
                else:
                    score = self._search(child, -side, probe_depth, alpha, beta, ply + 1, next_extension)
                if reduced and score > value:
                    score = self._search(child, -side, child_depth, alpha, beta, ply + 1, next_extension)
                if score > value:
                    value, best_move = score, move
                alpha = max(alpha, value)
                if alpha >= beta:
                    self.history[(side, move)] = self.history.get((side, move), 0) + depth * depth
                    self._remember_killer(ply, move)
                    break
        else:
            value = inf
            for idx, move in enumerate(ordered):
                child = apply_move(board, move)
                child_depth = local_depth - 1
                reduced = idx >= 4 and child_depth >= 2 and not move.captures and not move_promotes(board, move)
                probe_depth = child_depth - 1 if reduced else child_depth
                if self.config.use_pvs and idx > 0 and alpha < beta - 1.0:
                    score = self._search(child, -side, probe_depth, beta - 1.0, beta, ply + 1, next_extension)
                    if alpha < score < beta:
                        score = self._search(child, -side, probe_depth, alpha, beta, ply + 1, next_extension)
                else:
                    score = self._search(child, -side, probe_depth, alpha, beta, ply + 1, next_extension)
                if reduced and score < value:
                    score = self._search(child, -side, child_depth, alpha, beta, ply + 1, next_extension)
                if score < value:
                    value, best_move = score, move
                beta = min(beta, value)
                if alpha >= beta:
                    self.history[(side, move)] = self.history.get((side, move), 0) + depth * depth
                    self._remember_killer(ply, move)
                    break

        flag = "EXACT"
        if value <= alpha0:
            flag = "UPPER"
        elif value >= beta0:
            flag = "LOWER"
        if len(self.tt) >= self.config.tt_max_entries:
            self.tt.clear()
        self.tt[key] = TTEntry(depth=depth, value=value, flag=flag, best_move=best_move)
        return value

    @staticmethod
    def _instability(scores: list[float]) -> float:
        if len(scores) < 2:
            return 0.0
        return mean(abs(b - a) for a, b in zip(scores, scores[1:]))

    def _extract_pv(self, board: Board, side: int, first_move: Move, max_len: int) -> list[Move]:
        pv = [first_move]
        b = apply_move(board, first_move)
        s = -side
        for _ in range(max_len - 1):
            entry = self.tt.get((b, s, self.root_side))
            if not entry or not entry.best_move:
                break
            move = entry.best_move
            if move not in legal_moves(b, s):
                break
            pv.append(move)
            b = apply_move(b, move)
            s = -s
        return pv

    def choose(self, board: Board, side: int) -> SearchResult:
        start = monotonic()
        self.deadline = start + max(0.01, self.config.time_limit_ms / 1000)
        self.root_side = side
        self.nodes = 0
        if not self.config.persistent_tt:
            self.tt.clear()
        self.history.clear()
        self.killers.clear()

        moves = legal_moves(board, side)
        if not moves:
            return SearchResult(None, -MATE, 0, 0, 0.0, source="terminal")

        # ADAMS V3 layer 1: empirical opening knowledge learned from self-play.
        if self.config.use_opening_book and self.opening_book is not None:
            choice = self.opening_book.choose(board, side, min_visits=self.config.book_min_visits)
            if choice is not None:
                elapsed = (monotonic() - start) * 1000
                empirical = ((choice.wins + 0.5 * choice.draws) / max(1, choice.visits) - 0.5) * 200
                return SearchResult(
                    choice.move,
                    empirical,
                    0,
                    0,
                    elapsed,
                    [choice.move],
                    [],
                    source="opening_book",
                    extra={
                        "visits": choice.visits,
                        "wins": choice.wins,
                        "draws": choice.draws,
                        "losses": choice.losses,
                    },
                )

        # ADAMS V3 layer 2: exact retrograde knowledge for all built 1v1 states.
        if self.config.use_tablebase and self.tablebase is not None:
            exact = self.tablebase.lookup(board, side)
            if exact is not None and exact.best_move is not None:
                elapsed = (monotonic() - start) * 1000
                score = 0.0 if exact.outcome == 0 else exact.outcome * (MATE - exact.distance)
                return SearchResult(
                    exact.best_move, score, exact.distance, 0, elapsed, [exact.best_move], [],
                    source="exact_tablebase",
                    extra={"outcome": exact.outcome, "distance": exact.distance},
                )

        # ADAMS V3 layer 3: try to prove 3/4-piece endgames and persist proof.
        if self.config.use_endgame_db and self.endgame_db is not None:
            endgame_budget_ms = min(75, max(5, int(self.config.time_limit_ms * 0.15)))
            solved = self.endgame_db.solve(board, side, max_time_ms=endgame_budget_ms)
            if solved is not None and solved.best_move is not None:
                elapsed = (monotonic() - start) * 1000
                score = solved.outcome * (MATE - solved.distance)
                return SearchResult(
                    solved.best_move,
                    score,
                    solved.distance,
                    solved.nodes,
                    elapsed,
                    [solved.best_move],
                    [],
                    source="endgame_db",
                    extra={"outcome": solved.outcome, "distance": solved.distance, "proof_nodes": solved.nodes},
                )

        if len(moves) == 1:
            only = moves[0]
            elapsed = (monotonic() - start) * 1000
            return SearchResult(only, evaluate(apply_move(board, only), side, self.weights), 0, 0, elapsed, [only], [], source="forced")

        profiles = self._profile_candidates(board, side, moves)
        score_history: dict[Move, list[float]] = {p.move: [] for p in profiles}
        best_move = max(profiles, key=lambda p: p.static_score).move
        best_score = next(p.static_score for p in profiles if p.move == best_move)
        completed_depth = 0

        allocation_log: list[dict] = []
        for base_depth in range(1, self.config.max_depth + 1):
            iteration_scores: dict[Move, float] = {}
            iteration_meta: list[dict] = []
            try:
                latest = {
                    p.move: (score_history[p.move][-1] if score_history[p.move] else p.static_score)
                    for p in profiles
                }
                leader = max(latest.values()) if latest else 0.0

                def allocation_priority(profile: CandidateProfile) -> float:
                    hist = score_history[profile.move]
                    instability = self._instability(hist)
                    gap = leader - latest[profile.move]
                    uncertainty = instability + 0.18 * profile.volatility + 0.10 * profile.danger
                    # High score and high uncertainty near the leader get searched first.
                    return latest[profile.move] - 0.35 * gap + uncertainty

                ordered_profiles = sorted(profiles, key=allocation_priority, reverse=True)
                for rank, profile in enumerate(ordered_profiles):
                    hist = score_history[profile.move]
                    instability = self._instability(hist)
                    gap = leader - latest[profile.move]
                    uncertainty = instability + 0.18 * profile.volatility + 0.10 * profile.danger
                    extra = profile.extra_depth
                    focused = False
                    if self.config.use_cgra and base_depth >= self.config.focus_after_depth:
                        contender = gap <= self.config.confidence_margin + uncertainty
                        if rank < self.config.focus_candidates or contender:
                            extra = min(3, extra + 1)
                            focused = True
                    child = apply_move(board, profile.move)
                    depth = max(0, base_depth - 1 + extra)

                    if hist and self.config.use_aspiration:
                        window = self.config.aspiration_base + 0.35 * profile.volatility + 0.8 * instability
                        lo, hi = hist[-1] - window, hist[-1] + window
                        score = self._search(child, -side, depth, lo, hi, 1, self.config.extension_budget)
                        if score <= lo or score >= hi:
                            score = self._search(child, -side, depth, -inf, inf, 1, self.config.extension_budget)
                    else:
                        score = self._search(child, -side, depth, -inf, inf, 1, self.config.extension_budget)
                    iteration_scores[profile.move] = score
                    iteration_meta.append({
                        "rank": rank + 1,
                        "move": profile.move.to_json(),
                        "depth": depth,
                        "gap_before": round(gap, 2),
                        "uncertainty": round(uncertainty, 2),
                        "focused": focused,
                    })
            except SearchTimeout:
                break

            for move, score in iteration_scores.items():
                score_history[move].append(score)
            completed_depth = base_depth
            allocation_log = iteration_meta

            def robust(move: Move) -> float:
                hist = score_history[move]
                return hist[-1] - self.config.stability_penalty * self._instability(hist)

            best_move = max(iteration_scores, key=robust)
            best_score = iteration_scores[best_move]
            if best_score >= MATE - 100:
                break

        for p in profiles:
            if score_history[p.move]:
                p.latest_score = score_history[p.move][-1]
        profiles.sort(key=lambda p: (-inf if p.latest_score is None else p.latest_score), reverse=True)
        pv = self._extract_pv(board, side, best_move, max(2, completed_depth + 2))
        elapsed = (monotonic() - start) * 1000
        return SearchResult(
            best_move, best_score, completed_depth, self.nodes, elapsed, pv, profiles, source="search",
            extra={
                "root_allocator": "CGRA-v1" if self.config.use_cgra else "disabled",
                "allocation": allocation_log[:8],
                "tt_entries": len(self.tt),
                "pvs": self.config.use_pvs,
                "aspiration": self.config.use_aspiration,
            },
        )
