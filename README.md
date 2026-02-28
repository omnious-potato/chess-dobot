# 🤖 Chess playing robot with Dobit Magician ♟️

This project integrates computer vision, robotic control, and a chess engine to create a Dobot-powered chess-playing assistant. The robot can detect the current state of a chessboard using Data Matrix and a camera, process moves, and physically manipulate pieces using a Dobot Magician arm.

## ✨ Features

- 🎯 Uses **AprilTags**, **Data Matrix** and a webcam to detect the board state.
- 🎯 Supports optional **ArUco arm marker closed-loop correction** for robot XY alignment.
- 🧠 Integrates with **Stockfish** chess engine to play against the human.
- 🦾 Controls a **Dobot Magician** to move pieces on the board.
- 🔄 Handles **captures**, **promotions**, and **board scanning**.

---

## 🧰 Prerequisites

- Python ≥ 3.7  
- [Stockfish](https://stockfishchess.org/download/) chess engine (in the script assumed accessible at `/usr/bin/stockfish`)
- [libdmtx](https://github.com/dmtx/libdmtx) Data Matrix detection library
- Dobot Magician with proper drivers and connection established (usually `/dev/ttyUSB*`)
- Camera connected and accessible (default index: `3`)

### 🐍 Python Dependencies

Install required packages using pip:

```bash
pip install opencv-contrib-python apriltag numpy chess pydobot pylibdmtx
```

`opencv-contrib-python` is required for `cv2.aruco`.

### Closed-loop marker configuration (optional)

The runtime now supports visual-servo correction with:
- 4 board AprilTags (for board reference in image)
- 1 ArUco marker fixed on/with the actuator (with known XY offset to the suction tool)

Configuration is controlled via environment variables:

```bash
DOBOT_VSERVO_ENABLED=1
DOBOT_ARM_MARKER_ID=13
DOBOT_ARM_MARKER_OFFSET_X_MM=0
DOBOT_ARM_MARKER_OFFSET_Y_MM=0
DOBOT_VSERVO_JAC_STEP_MM=3
DOBOT_VSERVO_MAX_ITER=6
DOBOT_VSERVO_CONVERGENCE_PX=2.5
```

If ArUco detection is unavailable/fails, the code falls back to the original affine board-to-robot mapping.

### Standalone visual-servo calibration

You can calibrate marker-to-tool offset without starting the chess game:

```bash
python calibrate_visual_servo.py
# or with convenience wrapper
bash run_calibration.sh
```

What it does:
- connects only camera + Dobot
- visits multiple board squares
- estimates the XY correction needed to align the arm marker with board targets
- saves robust median offset to `vision_servo_calibration.json`
- shows guided UI for each sample:
  - step 1: **before correction** (predicted, actual, and error)
  - step 2: **after correction** (predicted, actual, and error)

In UI mode:
- Green cross/circle = predicted target position
- Red cross/circle = actual detected arm marker center
- Yellow arrow = error vector (actual -> predicted)
- Blocking mode keys: `Space`/`Enter` continue, `Q`/`Esc` abort

The game runtime (`main.py`) uses this file automatically as the starting point on startup.  
Environment variables still override file values if set explicitly.

Optional calibration env vars:

```bash
DOBOT_VSERVO_CALIBRATION_FILE=vision_servo_calibration.json
DOBOT_VSERVO_CALIB_SAMPLES="1,1;6,1;6,6;1,6;3,3;4,4"
DOBOT_VSERVO_CALIB_MAX_ITER=6
DOBOT_VSERVO_CALIB_HOVER_MM=15
DOBOT_VSERVO_UI_ENABLED=1
DOBOT_VSERVO_UI_BLOCKING=1
DOBOT_VSERVO_UI_WAIT_MS=1
DOBOT_VSERVO_UI_WINDOW="Visual Servo Calibration"
```

Wrapper examples:

```bash
bash run_calibration.sh --camera 0 --marker-id 13
bash run_calibration.sh --samples "1,1;3,3;6,6" --non-blocking-ui --wait-ms 30
```

### Allow serial port for dobot be accessed without superuser
This is optional (immediate alternative is launching python interpreter with superuser rights, which may be undesirable), process may differ depending on distributive (instructions are assumed for Debian/Ubuntu).

Check permissions of current user for the desires port (replace 'X' with the port that connected Dobot occupies)
```bash
ls -l /dev/ttyUSBX
```
If it's marked as being owned by the group ```dialout``` add current user to it, else consult your distributive documentation
```bash
sudo usermod -a -G dialout $USER
```
Source your shell and it should work
