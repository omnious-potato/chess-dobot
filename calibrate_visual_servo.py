import cv2
import apriltag
import numpy as np
import json
import os
import time
from datetime import datetime, timezone

from pydobot import Dobot


CAMERA_INDEX = int(os.getenv("DOBOT_CAMERA_INDEX", "3"))
MANUAL_MARKERS_FILE = os.getenv("DOBOT_MANUAL_MARKERS_FILE", "manual_markers.txt")
CALIBRATION_OUTPUT_FILE = os.getenv("DOBOT_VSERVO_CALIBRATION_FILE", "vision_servo_calibration.json")

ARM_MARKER_ID = int(os.getenv("DOBOT_ARM_MARKER_ID", "13"))
MARKER_MIN_AREA_PX = float(os.getenv("DOBOT_VSERVO_MIN_AREA_PX", "60"))
JACOBIAN_STEP_MM = float(os.getenv("DOBOT_VSERVO_JAC_STEP_MM", "3"))
HOVER_OFFSET_MM = float(os.getenv("DOBOT_VSERVO_CALIB_HOVER_MM", "15"))
MAX_ITERATIONS = int(os.getenv("DOBOT_VSERVO_CALIB_MAX_ITER", "6"))
CONVERGENCE_PX = float(os.getenv("DOBOT_VSERVO_CONVERGENCE_PX", "2.5"))
MAX_STEP_MM = float(os.getenv("DOBOT_VSERVO_MAX_STEP_MM", "6"))
UI_ENABLED = os.getenv("DOBOT_VSERVO_UI_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
UI_BLOCKING = os.getenv("DOBOT_VSERVO_UI_BLOCKING", "1").strip().lower() in {"1", "true", "yes", "on"}
UI_WAIT_MS = int(os.getenv("DOBOT_VSERVO_UI_WAIT_MS", "1"))
UI_WINDOW_NAME = os.getenv("DOBOT_VSERVO_UI_WINDOW", "Visual Servo Calibration")

BOARD_FRAME_CORNERS = np.array(
    [
        [0.0, 0.0],
        [8.0, 0.0],
        [8.0, 8.0],
        [0.0, 8.0],
    ],
    dtype=np.float32,
)

DEFAULT_SAMPLE_SQUARES = [
    (1, 1),
    (6, 1),
    (6, 6),
    (1, 6),
    (3, 3),
    (4, 4),
]


class CalibrationAborted(Exception):
    pass


def parse_sample_squares():
    raw = os.getenv("DOBOT_VSERVO_CALIB_SAMPLES", "").strip()
    if not raw:
        return list(DEFAULT_SAMPLE_SQUARES)

    squares = []
    for entry in raw.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        x_str, y_str = entry.split(",")
        squares.append((int(x_str), int(y_str)))
    return squares


def try_connecting_dobot():
    last_e = None
    for i in range(6):
        port = f"/dev/ttyUSB{i}"
        try:
            print(f"Trying to connect Dobot via {port}")
            device = Dobot(port=port)
            print(f"Connected on {port}")
            return device
        except Exception as e:
            last_e = e
    raise last_e


def get_coords_fromfile(path=MANUAL_MARKERS_FILE):
    board = []
    aux = []
    home = []

    with open(path, "r") as file:
        for i, line in enumerate(file):
            parts = line.split()
            if len(parts) < 3:
                continue
            xyz = [float(parts[0]), float(parts[1]), float(parts[2])]
            if i < 4:
                board.append(xyz)
            elif i < 8:
                aux.append(xyz)
            else:
                home = xyz

    return np.array(board), np.array(aux), tuple(home)


def get_board_to_robot_affine(src_pts, dst_pts):
    src_pts = np.asarray(src_pts, dtype=np.float32)
    dst_pts = np.asarray(dst_pts, dtype=np.float32)

    A = []
    Bx, By, Bz = [], [], []
    for (x, y), (X, Y, Z) in zip(src_pts, dst_pts):
        A.append([x, y, 1])
        Bx.append(X)
        By.append(Y)
        Bz.append(Z)

    A = np.array(A)
    coeffs_x, _, _, _ = np.linalg.lstsq(A, np.array(Bx), rcond=None)
    coeffs_y, _, _, _ = np.linalg.lstsq(A, np.array(By), rcond=None)
    coeffs_z, _, _, _ = np.linalg.lstsq(A, np.array(Bz), rcond=None)
    return np.vstack([coeffs_x, coeffs_y, coeffs_z])


def board_to_robot(board_affine, x, y):
    pt = np.array([x + 0.5, y + 0.5, 1.0], dtype=np.float32)
    result = board_affine @ pt
    return float(result[0]), float(result[1]), float(result[2])


def capture_image(cap):
    if not cap.isOpened():
        return None
    ok, frame = cap.read()
    return frame if ok else None


def detect_apriltags(frame, detector):
    gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return detector.detect(gray)


def detect_aruco_markers(frame, aruco_dict, aruco_detector, aruco_params):
    gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if aruco_detector is not None:
        corners, ids, _ = aruco_detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    return corners, ids


def detect_arm_marker_center(frame, aruco_dict, aruco_detector, aruco_params):
    corners, ids = detect_aruco_markers(frame, aruco_dict, aruco_detector, aruco_params)
    if ids is None:
        return None

    for marker_corners, marker_id in zip(corners, ids.flatten()):
        if int(marker_id) != ARM_MARKER_ID:
            continue
        pts = np.asarray(marker_corners, dtype=np.float32).reshape(-1, 2)
        if cv2.contourArea(pts) < MARKER_MIN_AREA_PX:
            continue
        return np.mean(pts, axis=0).astype(np.float32)
    return None


def _tag_center(tag):
    return np.mean(np.asarray(tag.corners, dtype=np.float32), axis=0)


def extract_board_corners(tags):
    if len(tags) < 4:
        raise ValueError("Need at least 4 board AprilTags")

    sorted_by_area = sorted(
        tags,
        key=lambda t: cv2.contourArea(np.asarray(t.corners, dtype=np.float32)),
        reverse=True,
    )
    board_tags = sorted_by_area[:4]

    centers = np.asarray([_tag_center(tag) for tag in board_tags], dtype=np.float32)
    y_order = np.argsort(centers[:, 1])
    top = y_order[:2]
    bottom = y_order[2:]

    top = top[np.argsort(centers[top, 0])]
    bottom = bottom[np.argsort(centers[bottom, 0])]

    ordered_tags = [
        board_tags[top[0]],
        board_tags[top[1]],
        board_tags[bottom[1]],
        board_tags[bottom[0]],
    ]

    board_center = np.mean(np.asarray([_tag_center(tag) for tag in ordered_tags]), axis=0)
    corners = []
    for tag in ordered_tags:
        tag_corners = np.asarray(tag.corners, dtype=np.float32)
        idx = np.argmin(np.linalg.norm(tag_corners - board_center, axis=1))
        corners.append(tag_corners[idx])

    return np.asarray(corners, dtype=np.float32)


def estimate_board_homography(frame, april_detector):
    tags = detect_apriltags(frame, april_detector)
    board_corners = extract_board_corners(tags)
    homography, _ = cv2.findHomography(BOARD_FRAME_CORNERS, board_corners, method=0)
    if homography is None:
        raise ValueError("Failed to estimate board homography")
    return homography


def project_board_point(board_xy, homography):
    point = np.array([board_xy[0], board_xy[1], 1.0], dtype=np.float32)
    projected = homography @ point
    if abs(float(projected[2])) < 1e-6:
        raise ValueError("Degenerate board homography")
    projected /= projected[2]
    return projected[:2].astype(np.float32)


def move_to(device, x, y, z, r=None, pause=0.1):
    if r is None:
        r = device.pose()[3]
    device.move_to(float(x), float(y), float(z), float(r), wait=True)
    time.sleep(pause)


def _to_point(px):
    return int(round(float(px[0]))), int(round(float(px[1])))


def _error_vector(marker_px, target_px):
    return target_px - marker_px


def _draw_feedback_overlay(frame, marker_px, target_px, square, stage_label):
    canvas = frame.copy()

    actual_pt = _to_point(marker_px)
    predicted_pt = _to_point(target_px)
    err = _error_vector(marker_px, target_px)
    err_norm = float(np.linalg.norm(err))

    # Predicted target point (board projection)
    cv2.drawMarker(canvas, predicted_pt, (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=22, thickness=2)
    cv2.circle(canvas, predicted_pt, 9, (0, 255, 0), 2)

    # Actual observed marker center
    cv2.drawMarker(canvas, actual_pt, (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=20, thickness=2)
    cv2.circle(canvas, actual_pt, 9, (0, 0, 255), 2)

    # Error vector from actual -> predicted
    cv2.arrowedLine(canvas, actual_pt, predicted_pt, (0, 255, 255), 2, tipLength=0.18)

    lines = [
        f"Stage: {stage_label}",
        f"Square: ({square[0]}, {square[1]})",
        f"Predicted(px): ({target_px[0]:.1f}, {target_px[1]:.1f})",
        f"Actual(px): ({marker_px[0]:.1f}, {marker_px[1]:.1f})",
        f"Error(px): dx={err[0]:.2f}, dy={err[1]:.2f}, norm={err_norm:.2f}",
    ]
    if UI_BLOCKING:
        lines.append("Keys: SPACE/ENTER=continue, Q/ESC=abort")

    y = 30
    for line in lines:
        cv2.putText(canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        y += 26

    return canvas, err, err_norm


def _show_feedback_frame(frame, marker_px, target_px, square, stage_label):
    if not UI_ENABLED:
        return True

    try:
        overlay, _, _ = _draw_feedback_overlay(frame, marker_px, target_px, square, stage_label)
        cv2.imshow(UI_WINDOW_NAME, overlay)
    except cv2.error as e:
        print(f"UI disabled (OpenCV display error): {e}")
        return True

    if not UI_BLOCKING:
        key = cv2.waitKey(max(1, UI_WAIT_MS)) & 0xFF
        if key in (ord("q"), 27):
            return False
        return True

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            return False
        if key in (ord(" "), 13, 10):
            return True


def capture_observation(cap, april_detector, aruco_dict, aruco_detector, aruco_params, board_pos):
    frame = capture_image(cap)
    if frame is None:
        return None, None, None

    marker_px = detect_arm_marker_center(frame, aruco_dict, aruco_detector, aruco_params)
    if marker_px is None:
        return frame, None, None

    try:
        board_h = estimate_board_homography(frame, april_detector)
    except ValueError:
        return frame, None, None

    target_px = project_board_point((board_pos[0] + 0.5, board_pos[1] + 0.5), board_h)
    return frame, marker_px, target_px


def capture_marker_only(cap, aruco_dict, aruco_detector, aruco_params):
    frame = capture_image(cap)
    if frame is None:
        return None
    return detect_arm_marker_center(frame, aruco_dict, aruco_detector, aruco_params)


def estimate_visual_jacobian_xy(device, cap, aruco_dict, aruco_detector, aruco_params):
    pose = device.pose()
    step = JACOBIAN_STEP_MM
    if step <= 0:
        return None

    base_marker = capture_marker_only(cap, aruco_dict, aruco_detector, aruco_params)
    if base_marker is None:
        return None

    move_to(device, pose[0] + step, pose[1], pose[2], pose[3])
    marker_x = capture_marker_only(cap, aruco_dict, aruco_detector, aruco_params)
    move_to(device, pose[0], pose[1], pose[2], pose[3])
    if marker_x is None:
        return None

    move_to(device, pose[0], pose[1] + step, pose[2], pose[3])
    marker_y = capture_marker_only(cap, aruco_dict, aruco_detector, aruco_params)
    move_to(device, pose[0], pose[1], pose[2], pose[3])
    if marker_y is None:
        return None

    col_x = (marker_x - base_marker) / step
    col_y = (marker_y - base_marker) / step
    jacobian = np.column_stack((col_x, col_y)).astype(np.float32)

    try:
        _ = np.linalg.inv(jacobian)
    except np.linalg.LinAlgError:
        return None
    return jacobian


def run_sample(device, cap, april_detector, aruco_dict, aruco_detector, aruco_params, board_affine, square):
    x, y, z = board_to_robot(board_affine, square[0], square[1])
    hover_z = z + HOVER_OFFSET_MM
    pose = device.pose()
    move_to(device, x, y, hover_z, pose[3])
    nominal_pose = device.pose()

    frame_before, marker_before, target_before = capture_observation(
        cap, april_detector, aruco_dict, aruco_detector, aruco_params, square
    )
    if marker_before is None or target_before is None:
        return None
    if not _show_feedback_frame(
        frame_before,
        marker_before,
        target_before,
        square,
        "Before correction",
    ):
        raise CalibrationAborted("User aborted during pre-correction preview")

    jacobian = estimate_visual_jacobian_xy(device, cap, aruco_dict, aruco_detector, aruco_params)
    if jacobian is None:
        return None

    inv_j = np.linalg.inv(jacobian)

    for _ in range(MAX_ITERATIONS):
        _, marker_px, target_px = capture_observation(
            cap, april_detector, aruco_dict, aruco_detector, aruco_params, square
        )
        if marker_px is None or target_px is None:
            break

        error = target_px - marker_px
        residual = float(np.linalg.norm(error))
        best_residual = residual
        if residual <= CONVERGENCE_PX:
            break

        delta = inv_j @ error
        delta = np.clip(delta, -MAX_STEP_MM, MAX_STEP_MM)

        pose = device.pose()
        move_to(device, pose[0] + delta[0], pose[1] + delta[1], hover_z, pose[3])

    frame_after, marker_after, target_after = capture_observation(
        cap, april_detector, aruco_dict, aruco_detector, aruco_params, square
    )
    if marker_after is None or target_after is None:
        return None
    if not _show_feedback_frame(
        frame_after,
        marker_after,
        target_after,
        square,
        "After correction",
    ):
        raise CalibrationAborted("User aborted during post-correction preview")

    pose_after = device.pose()
    correction = np.array([pose_after[0] - nominal_pose[0], pose_after[1] - nominal_pose[1]], dtype=np.float32)
    err_before = _error_vector(marker_before, target_before)
    err_after = _error_vector(marker_after, target_after)

    return {
        "board_square": [int(square[0]), int(square[1])],
        "nominal_xy_mm": [float(nominal_pose[0]), float(nominal_pose[1])],
        "correction_xy_mm": [float(correction[0]), float(correction[1])],
        "predicted_before_px": [float(target_before[0]), float(target_before[1])],
        "actual_before_px": [float(marker_before[0]), float(marker_before[1])],
        "error_before_px": [float(err_before[0]), float(err_before[1])],
        "error_before_norm_px": float(np.linalg.norm(err_before)),
        "predicted_after_px": [float(target_after[0]), float(target_after[1])],
        "actual_after_px": [float(marker_after[0]), float(marker_after[1])],
        "error_after_px": [float(err_after[0]), float(err_after[1])],
        "error_after_norm_px": float(np.linalg.norm(err_after)),
    }


def summarize_offset(sample_records):
    corrections = np.array([s["correction_xy_mm"] for s in sample_records], dtype=np.float32)
    if corrections.size == 0:
        return None, None
    median = np.median(corrections, axis=0)
    spread = np.std(corrections, axis=0)
    return median, spread


def main():
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco is unavailable. Install opencv-contrib-python.")

    sample_squares = parse_sample_squares()
    print(f"Calibration squares: {sample_squares}")

    board_coords = np.array(
        [
            [0.5, 0.5],
            [7.5, 0.5],
            [7.5, 7.5],
            [0.5, 7.5],
        ],
        dtype=np.float32,
    )
    dobot_corners, _, _ = get_coords_fromfile()
    if len(dobot_corners) != 4:
        raise RuntimeError("manual_markers.txt must contain 4 primary board corners first.")

    board_affine = get_board_to_robot_affine(board_coords, dobot_corners)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera index {CAMERA_INDEX}")

    april_detector = apriltag.Detector(apriltag.DetectorOptions(families="tag25h9"))
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    if hasattr(cv2.aruco, "ArucoDetector"):
        aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
        aruco_params = None
    else:
        aruco_detector = None
        aruco_params = cv2.aruco.DetectorParameters_create()

    device = try_connecting_dobot()
    device._after_fork = lambda: None
    device.speed(80, 80)

    sample_records = []
    try:
        pose = device.pose()
        safe_home = (pose[0], pose[1], pose[2] + 15, pose[3])
        move_to(device, *safe_home)

        for square in sample_squares:
            print(f"Calibrating square {square}...")
            record = run_sample(
                device,
                cap,
                april_detector,
                aruco_dict,
                aruco_detector,
                aruco_params,
                board_affine,
                square,
            )
            if record is None:
                print(f"  skipped {square}: marker/tag detection failed")
                continue
            sample_records.append(record)
            print(
                "  correction(mm)=({:.3f}, {:.3f}) error(px): {:.3f} -> {:.3f}".format(
                    record["correction_xy_mm"][0],
                    record["correction_xy_mm"][1],
                    record["error_before_norm_px"],
                    record["error_after_norm_px"],
                )
            )

        median, spread = summarize_offset(sample_records)
        if median is None:
            raise RuntimeError("Calibration failed: no valid sample squares were captured.")

        payload = {
            "version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "method": "auto_closed_loop_bootstrap",
            "camera_index": CAMERA_INDEX,
            "arm_marker_id": ARM_MARKER_ID,
            "marker_to_tool_offset_x_mm": float(median[0]),
            "marker_to_tool_offset_y_mm": float(median[1]),
            "offset_spread_x_mm": float(spread[0]),
            "offset_spread_y_mm": float(spread[1]),
            "jacobian_step_mm": float(JACOBIAN_STEP_MM),
            "convergence_px": float(CONVERGENCE_PX),
            "sample_count": len(sample_records),
            "samples": sample_records,
        }

        with open(CALIBRATION_OUTPUT_FILE, "w") as f:
            json.dump(payload, f, indent=2)

        print(f"\nSaved calibration to: {CALIBRATION_OUTPUT_FILE}")
        print(
            "Estimated marker->tool offset (mm): "
            f"X={payload['marker_to_tool_offset_x_mm']:.3f}, "
            f"Y={payload['marker_to_tool_offset_y_mm']:.3f}"
        )

    except CalibrationAborted as e:
        print(f"\nCalibration aborted: {e}")
    finally:
        try:
            pose = device.pose()
            move_to(device, pose[0], pose[1], pose[2] + 15, pose[3], pause=0.0)
        except Exception:
            pass
        if UI_ENABLED:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        try:
            cap.release()
        except Exception:
            pass
        try:
            device.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
