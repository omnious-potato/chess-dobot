#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_calibration.sh [options]

Options:
  --camera <index>            Camera index (default: 3)
  --marker-id <id>            ArUco marker ID on actuator (default: 13)
  --samples "<x,y;...>"       Sample squares (default: "1,1;6,1;6,6;1,6;3,3;4,4")
  --manual-markers <path>     Path to manual_markers.txt (default: ./manual_markers.txt)
  --output <path>             Calibration output JSON (default: ./vision_servo_calibration.json)
  --jac-step <mm>             Jacobian perturbation step in mm (default: 3)
  --hover <mm>                Hover offset in mm during calibration (default: 15)
  --max-iter <n>              Max correction iterations per sample (default: 6)
  --convergence <px>          Convergence threshold in px (default: 2.5)
  --max-step <mm>             Max XY correction per iteration in mm (default: 6)
  --min-area <px>             Minimum detected ArUco area in px (default: 60)
  --non-blocking-ui           Show UI but do not block per frame
  --headless                  Disable OpenCV UI overlays
  --wait-ms <ms>              UI wait in non-blocking mode (default: 1)
  --window <name>             OpenCV window title
  --python <bin>              Python executable (default: python3 then python)
  -h, --help                  Show this help

Examples:
  ./run_calibration.sh
  ./run_calibration.sh --camera 0 --marker-id 17
  ./run_calibration.sh --samples "1,1;3,3;6,6" --non-blocking-ui --wait-ms 30
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DOBOT_CAMERA_INDEX="${DOBOT_CAMERA_INDEX:-3}"
DOBOT_ARM_MARKER_ID="${DOBOT_ARM_MARKER_ID:-13}"
DOBOT_VSERVO_CALIB_SAMPLES="${DOBOT_VSERVO_CALIB_SAMPLES:-1,1;6,1;6,6;1,6;3,3;4,4}"
DOBOT_MANUAL_MARKERS_FILE="${DOBOT_MANUAL_MARKERS_FILE:-$SCRIPT_DIR/manual_markers.txt}"
DOBOT_VSERVO_CALIBRATION_FILE="${DOBOT_VSERVO_CALIBRATION_FILE:-$SCRIPT_DIR/vision_servo_calibration.json}"
DOBOT_VSERVO_JAC_STEP_MM="${DOBOT_VSERVO_JAC_STEP_MM:-3}"
DOBOT_VSERVO_CALIB_HOVER_MM="${DOBOT_VSERVO_CALIB_HOVER_MM:-15}"
DOBOT_VSERVO_CALIB_MAX_ITER="${DOBOT_VSERVO_CALIB_MAX_ITER:-6}"
DOBOT_VSERVO_CONVERGENCE_PX="${DOBOT_VSERVO_CONVERGENCE_PX:-2.5}"
DOBOT_VSERVO_MAX_STEP_MM="${DOBOT_VSERVO_MAX_STEP_MM:-6}"
DOBOT_VSERVO_MIN_AREA_PX="${DOBOT_VSERVO_MIN_AREA_PX:-60}"
DOBOT_VSERVO_UI_ENABLED="${DOBOT_VSERVO_UI_ENABLED:-1}"
DOBOT_VSERVO_UI_BLOCKING="${DOBOT_VSERVO_UI_BLOCKING:-1}"
DOBOT_VSERVO_UI_WAIT_MS="${DOBOT_VSERVO_UI_WAIT_MS:-1}"
DOBOT_VSERVO_UI_WINDOW="${DOBOT_VSERVO_UI_WINDOW:-Visual Servo Calibration}"
PYTHON_BIN="${PYTHON_BIN:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --camera)
      DOBOT_CAMERA_INDEX="$2"; shift 2 ;;
    --marker-id)
      DOBOT_ARM_MARKER_ID="$2"; shift 2 ;;
    --samples)
      DOBOT_VSERVO_CALIB_SAMPLES="$2"; shift 2 ;;
    --manual-markers)
      DOBOT_MANUAL_MARKERS_FILE="$2"; shift 2 ;;
    --output)
      DOBOT_VSERVO_CALIBRATION_FILE="$2"; shift 2 ;;
    --jac-step)
      DOBOT_VSERVO_JAC_STEP_MM="$2"; shift 2 ;;
    --hover)
      DOBOT_VSERVO_CALIB_HOVER_MM="$2"; shift 2 ;;
    --max-iter)
      DOBOT_VSERVO_CALIB_MAX_ITER="$2"; shift 2 ;;
    --convergence)
      DOBOT_VSERVO_CONVERGENCE_PX="$2"; shift 2 ;;
    --max-step)
      DOBOT_VSERVO_MAX_STEP_MM="$2"; shift 2 ;;
    --min-area)
      DOBOT_VSERVO_MIN_AREA_PX="$2"; shift 2 ;;
    --non-blocking-ui)
      DOBOT_VSERVO_UI_BLOCKING="0"; shift ;;
    --headless)
      DOBOT_VSERVO_UI_ENABLED="0"; shift ;;
    --wait-ms)
      DOBOT_VSERVO_UI_WAIT_MS="$2"; shift 2 ;;
    --window)
      DOBOT_VSERVO_UI_WINDOW="$2"; shift 2 ;;
    --python)
      PYTHON_BIN="$2"; shift 2 ;;
    -h|--help)
      usage
      exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "Error: python3/python not found in PATH." >&2
    exit 1
  fi
fi

if [[ ! -f "$DOBOT_MANUAL_MARKERS_FILE" ]]; then
  echo "Error: manual markers file not found: $DOBOT_MANUAL_MARKERS_FILE" >&2
  exit 1
fi

if [[ ! -f "$SCRIPT_DIR/calibrate_visual_servo.py" ]]; then
  echo "Error: calibrate_visual_servo.py not found in $SCRIPT_DIR" >&2
  exit 1
fi

export DOBOT_CAMERA_INDEX
export DOBOT_ARM_MARKER_ID
export DOBOT_VSERVO_CALIB_SAMPLES
export DOBOT_MANUAL_MARKERS_FILE
export DOBOT_VSERVO_CALIBRATION_FILE
export DOBOT_VSERVO_JAC_STEP_MM
export DOBOT_VSERVO_CALIB_HOVER_MM
export DOBOT_VSERVO_CALIB_MAX_ITER
export DOBOT_VSERVO_CONVERGENCE_PX
export DOBOT_VSERVO_MAX_STEP_MM
export DOBOT_VSERVO_MIN_AREA_PX
export DOBOT_VSERVO_UI_ENABLED
export DOBOT_VSERVO_UI_BLOCKING
export DOBOT_VSERVO_UI_WAIT_MS
export DOBOT_VSERVO_UI_WINDOW

echo "Running visual-servo calibration with:"
echo "  Python: $PYTHON_BIN"
echo "  Camera: $DOBOT_CAMERA_INDEX"
echo "  Marker ID: $DOBOT_ARM_MARKER_ID"
echo "  Samples: $DOBOT_VSERVO_CALIB_SAMPLES"
echo "  Manual markers: $DOBOT_MANUAL_MARKERS_FILE"
echo "  Output JSON: $DOBOT_VSERVO_CALIBRATION_FILE"
echo "  UI enabled/blocking: $DOBOT_VSERVO_UI_ENABLED/$DOBOT_VSERVO_UI_BLOCKING"

exec "$PYTHON_BIN" "$SCRIPT_DIR/calibrate_visual_servo.py"
