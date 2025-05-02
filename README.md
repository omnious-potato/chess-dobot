# 🤖 Chess playing robot with Dobit Magician ♟️

This project integrates computer vision, robotic control, and a chess engine to create a Dobot-powered chess-playing assistant. The robot can detect the current state of a chessboard using Data Matrix and a camera, process moves, and physically manipulate pieces using a Dobot Magician arm.

## ✨ Features

- 🎯 Uses **AprilTags**, **Data Matrix** and a webcam to detect the board state.
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
pip install opencv-python apriltag numpy chess pydobot pylibdmtx
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
