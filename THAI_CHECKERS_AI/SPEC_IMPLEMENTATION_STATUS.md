> V8 เพิ่มไฟล์ V4 และ `README_V8_MG400_V4_TH.md` แทนข้อจำกัดการตรวจฮอสเดิม; อุปกรณ์จริงยังต้องทดสอบ

# Implementation status — MG400 connection/status update

- Desktop Tkinter shows real TCP reachability and RobotMode at startup and in dashboard.
- IP is editable in `robot_connection.json` through settings; motion and diagnostics use the same configured IP.
- Read-only check communicates with ports 29999/30003 without EnableRobot/Move/DO.
- PhysicalMatch stage is displayed separately: UI-open preview versus real game started with timestamp.
- Retry of read-only connection and reattempt of preflight (without forcing unsafe start) are available in the UI.
- `board.npz`, `calib.npz`, `clear_pos.npz`, and the user-supplied V4 MG400 control source are included. Actual calibration requires onsite verification.
- Automated offline tests: 27 passed; desktop smoke screenshot regenerated.
- **Not verified against actual MG400 or live camera at the installation site.**
- Default site safety approvals remain false; live motion continues to be denied until responsible operator validates required hardware interlocks, vacuum feedback and geometry. Actual king handling is still restricted.
