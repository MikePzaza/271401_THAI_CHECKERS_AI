"""Pure state machine for a real physical 8x8 match.

The real board is authoritative. Game state changes only when the camera
confirms the HUMAN move or confirms physical completion of the AI move.
No robot hardware is imported here: tests are deterministic and offline.
"""
from __future__ import annotations
from dataclasses import dataclass

from engine import (
    BOARD_SIZE, Board, Move, apply_move, board_key, crown_skipped,
    _quiet_moves_for_piece, belongs, move_promotes,
)
from game_logic import CheckersGame, HUMAN, AI, coord_of, move_label, move_numbers, number_of


class MatchError(RuntimeError):
    pass


def signature(board: Board) -> dict[int, str]:
    """Physical board occupancy by color, independently of the king markers."""
    result = {}
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            cell = (BOARD_SIZE - 1 - r) * BOARD_SIZE + c + 1
            val = board[r][c]
            if val:
                result[cell] = 'black' if val > 0 else 'orange'
    return result


def king_signature(board: Board) -> set[int]:
    """Squares that must have a physical king (green center dot)."""
    return {((BOARD_SIZE - 1 - r) * BOARD_SIZE + c + 1) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
            if abs(board[r][c]) == 2}


def expected_kings_after(board: Board, move: Move) -> set[int]:
    """ตำแหน่งฮอสจริงที่กล้องควรเห็นหลังเดิน (ถ้าชนะทันทีตอนถึงแถวสุดท้าย ไม่บังคับวางฮอส จึงไม่มีจุดเขียวที่ช่องนั้น)"""
    kings = king_signature(apply_move(board, move))
    if crown_skipped(board, move):
        kings.discard(number_of(move.end))
    return kings


def require_kings(actual_kings, expected_board, label):
    if actual_kings is not None and set(actual_kings) != king_signature(expected_board):
        raise MatchError(f'{label}: ตำแหน่งฮอสที่กล้องเห็น {sorted(actual_kings)} '
                         f'ไม่ตรงกับสถานะเกม {sorted(king_signature(expected_board))}')


def require_same(actual: dict[int, str], expected: dict[int, str], label: str):
    if actual != expected:
        diffs = []
        for n in range(1, 37):
            if actual.get(n) != expected.get(n):
                diffs.append(f'{n}({actual.get(n, "-")} ≠ {expected.get(n, "-")})')
        raise MatchError(f'{label}: กล้องอ่านไม่ตรงกับกระดานเกม — ' + ', '.join(diffs[:12]) + (' ...' if len(diffs) > 12 else ''))


def _all_quiet_moves(board: Board, side: int) -> list[Move]:
    """คืนตาเดินธรรมดาทั้งหมดของฝ่าย side (ใช้ตรวจว่าผู้เล่นเผลอเดินธรรมดาตอนมีตาบังคับกินหรือไม่)"""
    moves: list[Move] = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            piece = board[r][c]
            if belongs(piece, side):
                moves.extend(_quiet_moves_for_piece(board, r, c, piece))
    return moves


@dataclass
class MatchState:
    stage: str = 'INITIAL'  # INITIAL, HUMAN, AI_READY, AI_IN_PROGRESS, VERIFY_AI, GAME_OVER, ERROR
    pending: Move | None = None
    robot_status: str = ''


class PhysicalMatch:
    def __init__(self):
        self.game = CheckersGame()
        self.state = MatchState()
        self.robot_status: dict[int, str] = {}

    def reset(self):
        self.__init__()

    def start(self, camera: dict[int, str], first_side=HUMAN, *, kings=None):
        if self.state.stage != 'INITIAL':
            raise MatchError('เกมกำลังดำเนินอยู่ กดเริ่มเกมใหม่ก่อน')
        require_same(camera, signature(self.game.board), 'เริ่มเกม')
        require_kings(kings, self.game.board, 'เริ่มเกม')
        if first_side not in (HUMAN, AI):
            raise MatchError('ฝ่ายเริ่มต้นไม่ถูกต้อง')
        self.game.side_to_move = first_side
        self.game.repetitions.clear()
        self.game.repetitions[board_key(self.game.board, first_side)] = 1
        self.state.stage = 'HUMAN' if first_side == HUMAN else 'AI_READY'

    def classify_human(self, camera: dict[int, str], *, kings=None) -> str:
        """unchanged / legal / illegal / ambiguous: checks occupancy without applying move."""
        if self.state.stage != 'HUMAN':
            return 'not_human_turn'
        if camera == signature(self.game.board) and (kings is None or set(kings) == king_signature(self.game.board)):
            return 'unchanged'
        candidates = [
            m for m in self.game.human_moves
            if signature(apply_move(self.game.board, m)) == camera
            and (kings is None or set(kings) == expected_kings_after(self.game.board, m))
        ]
        return 'legal' if len(candidates) == 1 else ('ambiguous' if candidates else 'illegal')

    def progress_human(self, camera: dict[int, str], *, kings=None) -> str:
        """
        ข้อความความคืบหน้าของตาคน (หมากสีดำ) ไว้แสดงใน popup ระหว่างตรวจ:
        ตากิน -> บอกว่า ① หมากดำไปถึงช่องปลายทางแล้วหรือยัง ② หมากส้มที่ถูกกินถูกหยิบออกครบหรือยัง
        (ทำลำดับไหนก็ได้ หรือยกออกแล้ววางทีเดียวก็ได้ — ผ่านเมื่อภาพสุดท้ายตรงกับตาที่ถูกกติกา)
        """
        if self.state.stage != 'HUMAN':
            return ''
        board = self.game.board
        cur_black = {sq for sq, col in camera.items() if col == 'black'}
        cur_orange = {sq for sq, col in camera.items() if col == 'orange'}
        best = None
        for m in self.game.human_moves:
            if not m.is_capture:
                continue
            nums = move_numbers(m)
            caps = sorted(number_of(c) for c in m.captures)
            black_ok = nums[-1] in cur_black and nums[0] not in cur_black
            gone = [c for c in caps if c not in cur_orange]
            score = (2 if black_ok else 0) + len(gone)
            if best is None or score > best[0]:
                best = (score, m, nums, caps, black_ok, gone)
        if best is None or best[0] == 0:
            return 'กำลังตรวจตาเดินของคน (หมากสีดำ) ...'
        _score, m, nums, caps, black_ok, gone = best
        left = [c for c in caps if c not in gone]
        l1 = f'① หมากดำ {nums[0]} → {nums[-1]}: ' + ('✓ เห็นแล้ว' if black_ok else '⏳ รอ')
        l2 = (f'② หยิบหมากส้มที่ถูกกินช่อง {caps} ออก: ' + ('✓ ครบแล้ว' if not left else f'⏳ ยังเหลือช่อง {left}'))
        return (f'กำลังตรวจตากิน {move_label(m)}\n{l1}\n{l2}\n'
                '(ทำลำดับไหนก็ได้ หรือหยิบออกแล้ววางทีเดียวก็ได้)')

    def diagnose_human(self, camera: dict[int, str], *, kings=None) -> tuple[str, str]:
        """
        วิเคราะห์สาเหตุอย่างละเอียดเมื่อตาคนยังไม่ผ่าน classify_human:
        คืนค่า (kind, message_th) โดย kind เป็นหนึ่งใน:
          - 'not_human_turn'
          - 'unchanged'
          - 'legal'
          - 'ambiguous'
          - 'incomplete_scan'          (หมากดำหายไปจากภาพกล้องเพราะมือบัง/กำลังถือหมาก/แสงสะท้อน หรือจำนวนหมากเกิน)
          - 'capture_removal_pending'  (ขยับหมากดำไปช่องปลายทางตากินแล้ว แต่ยังไม่ได้หยิบหมากส้มที่ถูกกินออก)
          - 'king_mismatch'            (เดินหมากถูกช่องตามกติกาแล้ว แต่สถานะจุดสีเขียวฮอสไม่ตรง)
          - 'must_capture'             (ติดกติกาบังคับกิน — ผู้เล่นขยับเดินธรรมดาทั้งที่มีตากิน)
          - 'incomplete_multijump'     (ต้องกินหลายต่อ แต่หยุดที่จุดพักกลางทาง)
          - 'illegal'                  (เดินผิดกติกาอื่น ๆ พร้อมแสดงรายการตาที่เดินได้)
        """
        status = self.classify_human(camera, kings=kings)
        if status == 'not_human_turn':
            return ('not_human_turn', 'ยังไม่ถึงตาคน')
        if status == 'unchanged':
            return ('unchanged', '')
        if status == 'legal':
            return ('legal', '')
        if status == 'ambiguous':
            opts = ' | '.join(move_label(m) for m in self.game.human_moves[:5])
            return ('ambiguous', f'ภาพกล้องตรงกับหลายเส้นทางกิน ({opts})')

        board = self.game.board
        moves = self.game.human_moves
        legal_list = ', '.join(move_label(m) for m in moves[:8])

        # 1) ตรวจสอบว่าตำแหน่งหมากตรงกับตาที่ถูกกติกาแล้ว แต่ติดเฉพาะเรื่องจุดเขียวฮอส (kings) หรือไม่
        pos_matches = [m for m in moves if signature(apply_move(board, m)) == camera]
        if len(pos_matches) == 1:
            m = pos_matches[0]
            want = expected_kings_after(board, m)
            seen_k = set(kings or ())
            end_sq = number_of(m.end)
            if move_promotes(board, m) and end_sq in want and end_sq not in seen_k:
                return (
                    'king_mismatch',
                    f'เดิน {move_label(m)} เข้าฮอสถูกต้องแล้ว! '
                    f'กรุณาเปลี่ยนหมากดำที่ช่อง {end_sq} เป็นตัวฮอส (ที่มีจุดสีเขียวตรงกลาง) เพื่อให้กล้องยืนยัน',
                )
            return (
                'king_mismatch',
                f'วางหมากตามตา {move_label(m)} ถูกช่องแล้ว แต่กล้องเห็นจุดเขียวฮอสที่ {sorted(seen_k)} '
                f'ไม่ตรงกับสถานะเกม {sorted(want)}\n'
                'กรุณาตรวจสติกเกอร์จุดเขียวบนตัวฮอส หรือขยับหมากเล็กน้อยเพื่อหลบแสงสะท้อน',
            )

        pre = signature(board)
        pre_black = {sq for sq, col in pre.items() if col == 'black'}
        pre_orange = {sq for sq, col in pre.items() if col == 'orange'}
        cur_black = {sq for sq, col in camera.items() if col == 'black'}
        cur_orange = {sq for sq, col in camera.items() if col == 'orange'}

        # 2) กรณีจำนวนหมากดำไม่เท่าเดิม (เช่น มือกำลังถือหมาก, มือบังหมาก, หรือแสงสะท้อนทำให้กล้องยังมองไม่เห็นหมากที่ช่องปลายทาง)
        if len(cur_black) < len(pre_black):
            missing = sorted(pre_black - cur_black)
            from_missing = [move_label(m) for m in moves if number_of(m.start) in missing]
            hint = f' (จากช่อง {missing[0]} เดินได้: {", ".join(from_missing[:6])})' if len(missing) == 1 and from_missing else ''
            return (
                'incomplete_scan',
                f'กล้องเห็นหมากดำ {len(cur_black)}/{len(pre_black)} ตัว (หายไปจากช่อง {missing}) — '
                f'หากวางหมากแล้ว กรุณาขยับหมากให้เข้ากลางช่องหรือหลบแสงสะท้อน{hint}',
            )
        if len(cur_black) > len(pre_black) or len(cur_orange) > len(pre_orange):
            return (
                'incomplete_scan',
                f'กล้องพบจำนวนหมากเกิน (ดำ {len(cur_black)}/{len(pre_black)}, ส้ม {len(cur_orange)}/{len(pre_orange)}) — '
                'กรุณานำหมากที่ถูกกินหรือวัตถุนอกเกมออกจากกรอบกระดาน',
            )

        # 3) กรณีผู้เล่นหยิบหมากส้มที่ถูกกินออกก่อนขยับหมากดำ (กำลังอยู่ในระหว่างทำตากิน)
        if cur_black == pre_black and len(cur_orange) < len(pre_orange) and any(m.is_capture for m in moves):
            removed_orange = sorted(pre_orange - cur_orange)
            matching_caps = [
                m for m in moves
                if m.is_capture and set(removed_orange).issubset({number_of(c) for c in m.captures})
            ]
            if matching_caps:
                labels = ', '.join(move_label(m) for m in matching_caps[:6])
                return (
                    'incomplete_scan',
                    f'หยิบหมากส้มช่อง {removed_orange} ออกแล้ว — กรุณาขยับหมากดำเพื่อเดินตากินให้ครบ: {labels}',
                )

        # 4) กรณีผู้เล่นขยับหมากดำไปยังช่องปลายทางของตากินถูกต้องแล้ว แต่ยังไม่ได้หยิบหมากส้มที่ถูกกินออก (หรือหยิบออกไม่ครบ)
        for m in moves:
            if not m.is_capture:
                continue
            after_sig = signature(apply_move(board, m))
            after_black = {sq for sq, col in after_sig.items() if col == 'black'}
            if cur_black == after_black:
                cap_squares = {number_of(c) for c in m.captures}
                still_there = sorted(cap_squares & cur_orange)
                wrongly_removed = sorted((pre_orange - cap_squares) - cur_orange)
                if still_there and not wrongly_removed:
                    return (
                        'capture_removal_pending',
                        f'เดินหมากดำ {move_label(m)} ถูกช่องแล้ว — '
                        f'กรุณาหยิบหมากส้มที่ถูกกินที่ช่อง {still_there} ออกจากกระดาน',
                    )

        # 5) กรณีต้องกินหลายต่อ (Multi-jump) แต่ผู้เล่นเพิ่งเดินไปหยุดที่ช่องกลางทาง
        for m in moves:
            if len(m.path) > 2:
                nums = move_numbers(m)
                start_sq = nums[0]
                for step_idx in range(1, len(nums) - 1):
                    mid_sq = nums[step_idx]
                    expected_mid_black = (pre_black - {start_sq}) | {mid_sq}
                    if cur_black == expected_mid_black:
                        return (
                            'incomplete_multijump',
                            f'ยังกินไม่ครบทุกต่อ! ตานี้ต้องกระโดดกินต่อเนื่องให้สุดทาง: {move_label(m)}\n'
                            f'(ตอนนี้หมากหยุดอยู่ที่ช่อง {mid_sq} กรุณากระโดดต่อไปยังช่อง {nums[-1]} และหยิบหมากส้มที่ถูกกินออกให้ครบ)',
                        )

        # 6) กรณีติดกติกา "บังคับกิน" (Mandatory Capture): มีตากินแต่ผู้เล่นไปขยับหมากเดินธรรมดา
        if any(m.is_capture for m in moves):
            for qm in _all_quiet_moves(board, HUMAN):
                if signature(apply_move(board, qm)) == camera:
                    return (
                        'must_capture',
                        f'ผิดกติกา (กติกาบังคับกิน!): คุณเดิน {move_label(qm)} แต่ในตานี้มีหมากที่สามารถกินได้\n'
                        f'โปรดย้ายหมากกลับไปช่อง {number_of(qm.start)} แล้วเดินตากิน: {legal_list}',
                    )
            moved_from = sorted(pre_black - cur_black)
            moved_to = sorted(cur_black - pre_black)
            if len(moved_from) == 1 and len(moved_to) == 1:
                return (
                    'must_capture',
                    f'ผิดกติกา (กติกาบังคับกิน!): คุณขยับหมากช่อง {moved_from[0]} → {moved_to[0]} '
                    f'แต่ตานี้บังคับต้องกินหมากก่อน!\n'
                    f'โปรดย้ายหมากกลับไปช่อง {moved_from[0]} แล้วเดินตากิน: {legal_list}',
                )

        # 7) กรณีขยับหมากผิดกติกาอื่น ๆ (เช่น เดินถอยหลัง, เดินผิดช่อง)
        moved_from = sorted(pre_black - cur_black)
        moved_to = sorted(cur_black - pre_black)
        if len(moved_from) == 1 and len(moved_to) == 1:
            src_sq = moved_from[0]
            dst_sq = moved_to[0]
            from_src = [move_label(m) for m in moves if number_of(m.start) == src_sq]
            if from_src:
                return (
                    'illegal',
                    f'เดินจากช่อง {src_sq} ไปช่อง {dst_sq} ไม่ได้!\n'
                    f'จากช่อง {src_sq} เดินได้เฉพาะ: {", ".join(from_src)}\n'
                    f'(ตาที่คนเดินได้ทั้งหมดในตานี้: {legal_list})',
                )
            return (
                'illegal',
                f'หมากที่ช่อง {src_sq} ไม่สามารถเดินได้ในตานี้ (คุณขยับ {src_sq} → {dst_sq})\n'
                f'โปรดย้ายหมากกลับช่อง {src_sq} | ตาที่คนเดินได้: {legal_list}',
            )

        return (
            'illegal',
            f'ตำแหน่งหมากบนกระดานไม่ตรงกับตาที่ถูกกติกา\n'
            f'ตาที่คนเดินได้ในตานี้: {legal_list}\n'
            f'โปรดจัดหมากกลับตำแหน่งเดิมหรือเดินตามรายการข้างต้น',
        )

    def accept_human(self, camera: dict[int, str], *, kings=None):
        if self.state.stage != 'HUMAN':
            raise MatchError('ยังไม่ถึงตาคน')
        require_pre = signature(self.game.board)
        if camera == require_pre and (kings is None or set(kings) == king_signature(self.game.board)):
            raise MatchError('ยังไม่พบการเปลี่ยนตำแหน่งหมากจากตาก่อน')
        possible = [
            m for m in self.game.human_moves
            if signature(apply_move(self.game.board, m)) == camera
            and (kings is None or set(kings) == expected_kings_after(self.game.board, m))
        ]
        if len(possible) == 0:
            _kind, msg = self.diagnose_human(camera, kings=kings)
            raise MatchError(msg or 'ไม่พบตาเดินที่ถูกกติกาตรงกับภาพกล้อง ตรวจหมากที่ถูกกินและแสง')
        if len(possible) > 1:
            raise MatchError('ภาพกล้องตรงกับหลายเส้นทางกิน ต้องยืนยันเส้นทางให้ชัดก่อน')
        move = possible[0]
        entry = self.game.play_human(move_label(move))
        if self.game.status != 'playing':
            self.state.stage = 'GAME_OVER'
        else:
            self.state.stage = 'AI_READY'  # AI worker will select move separately
        return entry

    def stage_ai_move(self, move: Move):
        if self.state.stage != 'AI_READY' or self.state.pending is not None:
            raise MatchError('ยังไม่พร้อมรับตา AI')
        from engine import legal_moves
        if move not in legal_moves(self.game.board, AI):
            raise MatchError('AI ส่งตาผิดกติกา')
        self.state.pending = move

    def require_board_ready(self, camera: dict[int, str], *, kings=None):
        """Read-only safety preflight; no state transition before operator approval."""
        require_same(camera, signature(self.game.board), 'ก่อนสั่ง MG400')
        require_kings(kings, self.game.board, 'ก่อนสั่ง MG400')

    def before_robot(self, camera: dict[int, str], *, kings=None):
        if self.state.stage != 'AI_READY' or self.state.pending is None:
            raise MatchError('ไม่มีตา AI ที่พร้อมเคลื่อน')
        require_same(camera, signature(self.game.board), 'ก่อนสั่ง MG400')
        require_kings(kings, self.game.board, 'ก่อนสั่ง MG400')
        self.state.stage = 'AI_IN_PROGRESS'

    def after_robot(self, status: str):
        if self.state.stage != 'AI_IN_PROGRESS':
            raise MatchError('ไม่ได้อยู่ในขั้นตอนหุ่นยนต์ทำงาน')
        self.state.robot_status = status
        self.state.stage = 'VERIFY_AI'

    def robot_failed(self, reason: str):
        self.state.stage = 'ERROR'
        self.state.robot_status = f'ERROR: {reason}'

    def verify_ai(self, camera: dict[int, str], *, kings=None):
        if self.state.stage != 'VERIFY_AI' or self.state.pending is None:
            raise MatchError('ยังไม่มีผลตา AI ที่ต้องตรวจกล้อง')
        expected = apply_move(self.game.board, self.state.pending)
        require_same(camera, signature(expected), 'หลัง AI เดิน')
        want = expected_kings_after(self.game.board, self.state.pending)
        if kings is not None and set(kings) != want:
            raise MatchError(f'หลัง AI เดิน: ตำแหน่งฮอสที่กล้องเห็น {sorted(kings)} '
                             f'ไม่ตรงกับสถานะเกม {sorted(want)}')
        entry = self.game.play_ai(self.state.pending)
        self.robot_status[entry.ply] = self.state.robot_status + ' / CAMERA VERIFIED'
        self.state.pending = None
        self.state.robot_status = ''
        self.state.stage = 'GAME_OVER' if self.game.status != 'playing' else 'HUMAN'
        return entry

    def move_text(self):
        return '' if self.state.pending is None else move_label(self.state.pending)
