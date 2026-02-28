import cv2
import apriltag
import numpy as np
import json
import chess
import chess.engine
import time
import os
from dataclasses import dataclass

import threading    

from pydobot import Dobot
from segment import segment_image

CAMERA_INDEX = 3
VISION_SERVO_CALIBRATION_FILE = os.getenv("DOBOT_VSERVO_CALIBRATION_FILE", "vision_servo_calibration.json")


@dataclass(frozen=True)
class VisionServoConfig:
    enabled: bool = True
    arm_marker_id: int = 13
    aruco_dict_id: int = cv2.aruco.DICT_4X4_50 if hasattr(cv2, "aruco") else -1
    marker_min_area_px: float = 60.0
    jacobian_step_mm: float = 3.0
    max_iterations: int = 6
    convergence_px: float = 2.5
    min_improvement_px: float = 0.15
    max_step_mm: float = 6.0
    min_jacobian_det: float = 1e-3
    marker_to_tool_offset_x_mm: float = 0.0
    marker_to_tool_offset_y_mm: float = 0.0


def _env_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_saved_vision_servo_calibration(path=VISION_SERVO_CALIBRATION_FILE):
    try:
        with open(path, "r") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return {}
        return payload
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _env_or_default(name, default_value):
    value = os.getenv(name)
    if value is None:
        return default_value
    return value


def _load_vision_servo_config():
    saved = _load_saved_vision_servo_calibration()
    saved_marker_id = int(saved.get("arm_marker_id", 13))
    saved_offset_x = float(saved.get("marker_to_tool_offset_x_mm", 0))
    saved_offset_y = float(saved.get("marker_to_tool_offset_y_mm", 0))
    saved_jac_step = float(saved.get("jacobian_step_mm", 3))
    saved_convergence_px = float(saved.get("convergence_px", 2.5))

    return VisionServoConfig(
        enabled=_env_bool("DOBOT_VSERVO_ENABLED", True),
        arm_marker_id=int(_env_or_default("DOBOT_ARM_MARKER_ID", str(saved_marker_id))),
        marker_min_area_px=float(_env_or_default("DOBOT_VSERVO_MIN_AREA_PX", "60")),
        jacobian_step_mm=float(_env_or_default("DOBOT_VSERVO_JAC_STEP_MM", str(saved_jac_step))),
        max_iterations=int(_env_or_default("DOBOT_VSERVO_MAX_ITER", "6")),
        convergence_px=float(_env_or_default("DOBOT_VSERVO_CONVERGENCE_PX", str(saved_convergence_px))),
        min_improvement_px=float(_env_or_default("DOBOT_VSERVO_MIN_IMPROVEMENT_PX", "0.15")),
        max_step_mm=float(_env_or_default("DOBOT_VSERVO_MAX_STEP_MM", "6")),
        marker_to_tool_offset_x_mm=float(_env_or_default("DOBOT_ARM_MARKER_OFFSET_X_MM", str(saved_offset_x))),
        marker_to_tool_offset_y_mm=float(_env_or_default("DOBOT_ARM_MARKER_OFFSET_Y_MM", str(saved_offset_y))),
    )


VISION_SERVO_CONFIG = _load_vision_servo_config()
BOARD_FRAME_CORNERS = np.array(
    [
        [0.0, 0.0],
        [8.0, 0.0],
        [8.0, 8.0],
        [0.0, 8.0],
    ],
    dtype=np.float32,
)


def get_board_to_robot_affine(src_pts, dst_pts):
    """
    Calculates a 2D->3D affine transformation from board coords to robot coords.
    src_pts: Nx2 (e.g., board squares like [[0.5, 0.5], ..., [7.5, 7.5]])
    dst_pts: Nx3 (e.g., measured Dobot coordinates like [[X, Y, Z], ...])
    Returns: 3x3 affine matrix A, where A @ [x, y, 1] = [X, Y, Z]
    """
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
    Bx = np.array(Bx)
    By = np.array(By)
    Bz = np.array(Bz)

    # Solve A @ coeffs = B for X, Y, Z independently
    coeffs_x, _, _, _ = np.linalg.lstsq(A, Bx, rcond=None)
    coeffs_y, _, _, _ = np.linalg.lstsq(A, By, rcond=None)
    coeffs_z, _, _, _ = np.linalg.lstsq(A, Bz, rcond=None)

    # Compose into a single 3x3 matrix
    transform = np.vstack([coeffs_x, coeffs_y, coeffs_z])
    return transform


# Initialize Dobot
def try_connecting_dobot():
    last_e = None

    for i in range(6):
        port = f'/dev/ttyUSB{i}'
        try:
            print(f'Trying to connect Dobot via USB{i}')

            dobot = Dobot(port=port)
            print(f'Connected on port {i}')

            return dobot
        except Exception as e:
            last_e = e

    raise last_e

dobot = try_connecting_dobot()
dobot._after_fork = lambda: None

dobot.speed(100, 100)

# Initialize chess engine (Stockfish)
engine = chess.engine.SimpleEngine.popen_uci("/usr/bin/stockfish") 

# Board state file
BOARD_STATE_FILE = "board_state.json"

# Camera setup
try:
    cap = cv2.VideoCapture(CAMERA_INDEX)
except Exception as e:
    print(f'Exception occured while opening camera: {e}')

# AprilTag detector
options = apriltag.DetectorOptions(families='tag25h9')
apriltag_detector = apriltag.Detector(options)

if hasattr(cv2, "aruco"):
    aruco_dictionary = cv2.aruco.getPredefinedDictionary(VISION_SERVO_CONFIG.aruco_dict_id)
    if hasattr(cv2.aruco, "ArucoDetector"):
        aruco_detector = cv2.aruco.ArucoDetector(aruco_dictionary, cv2.aruco.DetectorParameters())
    else:
        aruco_detector = None
        aruco_detector_params = cv2.aruco.DetectorParameters_create()
else:
    aruco_dictionary = None
    aruco_detector = None
    aruco_detector_params = None


# Global coordinates setup
line_count = 4
def get_coords_fromfile(path='manual_markers.txt'): 
    """Parses text file that expects to have 9 lines (4 for main boars + 4 for auxillary board + 1 home position)"""
    A = []
    B = []
    home = []

    with open(path, "r") as file:

        for i, line in enumerate(file):
            res = line.split()
            if i < 4:
                A.append([float(res[0]), float(res[1]), float(res[2])])
            elif i < 8:
                B.append([float(res[0]), float(res[1]), float(res[2])])
            else: 
                home = [float(res[0]), float(res[1]), float(res[2])]
            
    dobot_corners = np.array(A)
    aux_dobot_corners = np.array(B)

    return dobot_corners, aux_dobot_corners, tuple(home)
        


pos = dobot.pose()
home_position = (pos[0], pos[1], pos[2] + 15)

dobot_corners, aux_dobot_corners, dummy = get_coords_fromfile()
z_axis_limits = [home_position[2]- 8, home_position[2] - 3, home_position[2] - 8, home_position[2] - 3]

print(dobot_corners)
print(aux_dobot_corners)

board_coords = np.array([
    [0.5, 0.5],
    [7.5, 0.5],
    [7.5, 7.5],
    [0.5, 7.5]
])

aux_board_coords = np.array([
    [0.5, 0.5],    # Top-left
    [2.5, 0.5],     # Top-right
    [2.5, 6.5],    # Bottom-right
    [0.5, 6.5],    # Bottom-left
])

# Calculate homography matrices
# Get full 3D positions from file
dobot_corners_3d, aux_dobot_corners_3d, home_position = get_coords_fromfile()

# Compute affine transforms from board space (2D) to robot space (3D)
board_affine = get_board_to_robot_affine(board_coords, dobot_corners_3d)
aux_board_affine = get_board_to_robot_affine(aux_board_coords, aux_dobot_corners_3d)

# Global variables for tracking
captured_pieces = {}
aux_board_occupancy = [[False for _ in range(7)] for _ in range(3)]  # 3x7 auxiliary board
suction_delay = 0.5  # Delay for suction to activate/deactivate
move_delay = 0.1  # Delay between movements

def capture_image(device):
    """Captures an image from the camera."""
    if not device.isOpened():
        print(f"Capture device isn't open!")
        return None
    ret, frame = device.read()

    if ret:
        cv2.imwrite("./board_example.jpg", frame )


    return frame if ret else None

def detect_apriltags(img):
    """Detects AprilTags in the image."""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return apriltag_detector.detect(img)


def detect_aruco_markers(img):
    """Detects ArUco markers and returns (corners, ids)."""
    if aruco_dictionary is None:
        return [], None

    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if aruco_detector is not None:
        corners, ids, _ = aruco_detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray,
            aruco_dictionary,
            parameters=aruco_detector_params,
        )

    return corners, ids


def detect_arm_marker_center(img):
    """Returns the configured arm marker center in pixel coordinates, if visible."""
    corners, ids = detect_aruco_markers(img)
    if ids is None:
        return None

    for marker_corners, marker_id in zip(corners, ids.flatten()):
        if int(marker_id) != VISION_SERVO_CONFIG.arm_marker_id:
            continue

        pts = np.asarray(marker_corners, dtype=np.float32).reshape(-1, 2)
        if cv2.contourArea(pts) < VISION_SERVO_CONFIG.marker_min_area_px:
            continue
        return np.mean(pts, axis=0).astype(np.float32)

    return None


def _tag_center(tag):
    return np.mean(np.asarray(tag.corners, dtype=np.float32), axis=0)


def _extract_board_corners(tags):
    """
    Derives board corner pixels from the four board AprilTags.
    Returns corners ordered as TL, TR, BR, BL in image pixels.
    """
    if len(tags) < 4:
        raise ValueError("Could not detect enough AprilTags for board alignment")

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
        board_tags[top[0]],     # TL
        board_tags[top[1]],     # TR
        board_tags[bottom[1]],  # BR
        board_tags[bottom[0]],  # BL
    ]

    board_center = np.mean(np.asarray([_tag_center(tag) for tag in ordered_tags]), axis=0)
    corners = []
    for tag in ordered_tags:
        tag_corners = np.asarray(tag.corners, dtype=np.float32)
        idx = np.argmin(np.linalg.norm(tag_corners - board_center, axis=1))
        corners.append(tag_corners[idx])

    return np.asarray(corners, dtype=np.float32)


def _estimate_board_to_image_homography(frame):
    tags = detect_apriltags(frame)
    board_corners = _extract_board_corners(tags)
    homography, _ = cv2.findHomography(BOARD_FRAME_CORNERS, board_corners, method=0)
    if homography is None:
        raise ValueError("Failed to compute board homography from AprilTags")
    return homography, board_corners


def _project_board_point_to_image(board_xy, homography):
    point = np.array([board_xy[0], board_xy[1], 1.0], dtype=np.float32)
    projected = homography @ point
    if abs(float(projected[2])) < 1e-6:
        raise ValueError("Degenerate board homography projection")
    projected /= projected[2]
    return projected[:2].astype(np.float32)


def _capture_marker_and_target_pixels(board_pos):
    frame = capture_image(cap)
    if frame is None:
        return None, None

    marker_px = detect_arm_marker_center(frame)
    if marker_px is None:
        return None, None

    try:
        board_h, _ = _estimate_board_to_image_homography(frame)
    except ValueError:
        return None, None

    target_px = _project_board_point_to_image(
        (board_pos[0] + 0.5, board_pos[1] + 0.5),
        board_h,
    )
    return marker_px, target_px


def _capture_marker_pixels():
    frame = capture_image(cap)
    if frame is None:
        return None
    return detect_arm_marker_center(frame)


def _estimate_visual_jacobian_xy():
    """Estimates local marker pixel response wrt robot XY movement."""
    step = float(VISION_SERVO_CONFIG.jacobian_step_mm)
    if step <= 0:
        return None

    pose = dobot.pose()
    base_marker = _capture_marker_pixels()
    if base_marker is None:
        return None

    move_to_position(pose[0] + step, pose[1], pose[2], pose[3])
    marker_x = _capture_marker_pixels()
    move_to_position(pose[0], pose[1], pose[2], pose[3])
    if marker_x is None:
        return None

    move_to_position(pose[0], pose[1] + step, pose[2], pose[3])
    marker_y = _capture_marker_pixels()
    move_to_position(pose[0], pose[1], pose[2], pose[3])
    if marker_y is None:
        return None

    col_x = (marker_x - base_marker) / step
    col_y = (marker_y - base_marker) / step
    jacobian = np.column_stack((col_x, col_y)).astype(np.float32)

    if abs(float(np.linalg.det(jacobian))) < VISION_SERVO_CONFIG.min_jacobian_det:
        return None

    return jacobian


def _closed_loop_align_board_square(board_pos, hover_xyz):
    """
    Runs a visual-servo XY refinement against board AprilTags + arm ArUco marker.
    Returns corrected (x, y) and leaves robot at hover_z.
    """
    if not VISION_SERVO_CONFIG.enabled or aruco_dictionary is None:
        return hover_xyz[0], hover_xyz[1]

    jacobian = _estimate_visual_jacobian_xy()
    if jacobian is None:
        return hover_xyz[0], hover_xyz[1]

    try:
        jacobian_inv = np.linalg.inv(jacobian)
    except np.linalg.LinAlgError:
        return hover_xyz[0], hover_xyz[1]

    marker_to_tool_mm = np.array(
        [
            VISION_SERVO_CONFIG.marker_to_tool_offset_x_mm,
            VISION_SERVO_CONFIG.marker_to_tool_offset_y_mm,
        ],
        dtype=np.float32,
    )
    marker_to_tool_px = jacobian @ marker_to_tool_mm

    best_error = None
    for _ in range(VISION_SERVO_CONFIG.max_iterations):
        marker_px, target_px = _capture_marker_and_target_pixels(board_pos)
        if marker_px is None or target_px is None:
            break

        error_px = target_px - (marker_px + marker_to_tool_px)
        error_norm = float(np.linalg.norm(error_px))
        if error_norm <= VISION_SERVO_CONFIG.convergence_px:
            break

        if best_error is not None and (best_error - error_norm) < VISION_SERVO_CONFIG.min_improvement_px:
            break
        best_error = error_norm

        delta_mm = jacobian_inv @ error_px
        delta_mm = np.clip(
            delta_mm,
            -VISION_SERVO_CONFIG.max_step_mm,
            VISION_SERVO_CONFIG.max_step_mm,
        )

        pose = dobot.pose()
        move_to_position(
            float(pose[0] + delta_mm[0]),
            float(pose[1] + delta_mm[1]),
            hover_xyz[2],
            pose[3],
        )

    pose = dobot.pose()
    return float(pose[0]), float(pose[1])

def board_to_robot(x, y):
    """Convert board (x, y) to robot (X, Y, Z) using 3D affine transform"""
    pt = np.array([x + 0.5, y + 0.5, 1.0])
    res = board_affine @ pt
    return tuple(res)

def aux_board_to_robot(x, y):
    """Convert auxiliary board (x, y) to robot (X, Y, Z) using 3D affine transform"""
    pt = np.array([x + 0.5, y + 0.5, 1.0])
    res = aux_board_affine @ pt
    return tuple(res)


def board_to_robot_refined(board_pos, hover_offset=15.0):
    """
    Returns robot XYZ for a board square after optional closed-loop XY correction.
    The robot is moved to hover pose first, then refined in place.
    """
    x, y, z = board_to_robot(*board_pos)
    hover_xyz = (float(x), float(y), float(z + hover_offset))
    move_to_position(*hover_xyz)
    refined_x, refined_y = _closed_loop_align_board_square(board_pos, hover_xyz)
    return refined_x, refined_y, float(z)


def get_aux_coord():
    """Finds and returns coordinates for the next free auxiliary board slot."""
    for col in range(3):
        for row in range(7):
            if not aux_board_occupancy[col][row]:
                aux_board_occupancy[col][row] = True
                return aux_board_to_robot(col, row)
    raise Exception("Auxiliary board is full!")

dobot_lock = threading.Lock()

def move_to_position(x, y, z, r=0):
    """Moves Dobot to the specified position."""
    dobot.move_to(x, y, z, r, wait=True)
    time.sleep(move_delay)

def exec_move(from_pos, to_pos, isCapture=False, promotion=None):
    """
    Executes a chess move with the Dobot Magician.
    
    Args:
        from_pos: (x, y) board coordinates of source position
        to_pos: (x, y) board coordinates of destination position
        isCapture: Whether the move is a capture
        promotion: Piece type to promote to (if any)
    """
    global aux_board_occupancy

    z_low_main = z_axis_limits[0]
    z_high_main = z_axis_limits[1]
    z_low_aux = z_axis_limits[2]
    z_high_aux = z_axis_limits[3]

    # Handle capture (move captured piece to auxiliary board first)
    if isCapture:
        # Get free slot in auxiliary board
        aux_x, aux_y, aux_z = get_aux_coord()
        
        # Move to capture position and pick up piece
        cap_x, cap_y, cap_z = board_to_robot_refined(to_pos, hover_offset=10)
        move_to_position(cap_x, cap_y, cap_z - 8)
        dobot.suck(True)
        time.sleep(suction_delay)
        move_to_position(cap_x, cap_y, cap_z + 10)
        
        # Move to auxiliary board and drop piece
        move_to_position(aux_x, aux_y, aux_z + 10)
        move_to_position(aux_x, aux_y, aux_z - 8)
        dobot.suck(False)
        time.sleep(suction_delay)
        move_to_position(aux_x, aux_y, aux_z + 10)
    
    # Handle promotion
    if promotion:
        # Get promotion piece from auxiliary board
        aux_x, aux_y, aux_z = get_aux_coord()
        
        # Move to auxiliary board and pick up promotion piece
        move_to_position(aux_x, aux_y, z_high_aux)
        move_to_position(aux_x, aux_y, z_low_aux)
        dobot.suck(True)
        time.sleep(suction_delay)
        move_to_position(aux_x, aux_y, z_high_aux)
        
        # Move to destination and drop promotion piece
        to_x, to_y, _ = board_to_robot_refined(to_pos, hover_offset=15)
        move_to_position(to_x, to_y, z_high_main)
        move_to_position(to_x, to_y, z_low_main)
        dobot.suck(False)
        time.sleep(suction_delay)
        move_to_position(to_x, to_y, z_high_main)
        
        # Move original piece to auxiliary board
        from_x, from_y, _ = board_to_robot_refined(from_pos, hover_offset=15)
        move_to_position(from_x, from_y, z_high_main)
        move_to_position(from_x, from_y, z_low_main)
        dobot.suck(True)
        time.sleep(suction_delay)
        move_to_position(from_x, from_y, z_high_main)
        
        # Move to auxiliary board and drop original piece
        move_to_position(aux_x, aux_y, z_high_aux)
        move_to_position(aux_x, aux_y, z_low_aux)
        dobot.suck(False)
        time.sleep(suction_delay)
        move_to_position(aux_x, aux_y, z_high_aux)
        
        return
    
    # Standard move
    # Move to source position and pick up piece
    from_x, from_y, from_z = board_to_robot_refined(from_pos, hover_offset=15)
    move_to_position(from_x, from_y, from_z)
    dobot.suck(True)
    time.sleep(suction_delay)
    move_to_position(from_x, from_y, from_z + 15)
    
    # Move to destination position and drop piece
    to_x, to_y, to_z = board_to_robot_refined(to_pos, hover_offset=15)
    move_to_position(to_x, to_y, to_z)
    dobot.suck(False)
    time.sleep(suction_delay)
    move_to_position(to_x, to_y, to_z + 15)
    
    # Return to home position
    move_to_position(*home_position)

def scan_board():
    """Scans the current board state using computer vision."""
    img = capture_image(cap)
    if img is None:
        raise Exception("Failed to capture image")
    
    # Detect AprilTags to align the board
    _, board_markers = _estimate_board_to_image_homography(img)
    
    # Warp the perspective to get a clean board image
    board_size = 800
    dest_points = np.array([
        [0, 0],
        [board_size - 1, 0],
        [board_size - 1, board_size - 1],
        [0, board_size - 1],
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(board_markers, dest_points)
    warped = cv2.warpPerspective(img, M, (board_size, board_size))
    
    cv2.imwrite("./board_warped.jpg", warped)
    
    # Segment the image into squares and decode DataMatrix codes
    squares = segment_image(warped)
    #print(squares)
    
    board_state = [None] * 64
    
    for i, square_img in enumerate(squares):
        if len(square_img) == 0:
            piece_code = '.'
        else:
            piece_code = square_img.data.decode('ascii')
        
        board_state[i] = piece_code
    
    return board_state

def initialize_board_state():
    """Initializes or resets the board state."""
    board = chess.Board()
    save_board_state(board)
    return board

def load_board_state():
    """Loads the board state from file."""
    try:
        with open(BOARD_STATE_FILE, 'r') as f:
            fen = json.load(f)['fen']
            return chess.Board(fen)
    except (FileNotFoundError, json.JSONDecodeError):
        return initialize_board_state()

def save_board_state(board):
    """Saves the board state to file."""
    with open(BOARD_STATE_FILE, 'w') as f:
        json.dump({'fen': board.fen()}, f)

def validate_human_move(board, move_uci):
    """Validates a human move."""
    try:
        move = chess.Move.from_uci(move_uci)
        return move in board.legal_moves
    except:
        return False

def game_loop():
    """Main game loop handling chess logic and Dobot movements."""
    board = load_board_state()
    
    while not board.is_game_over():
        print("\nCurrent board:")
        print(board)
        
        try:
            scan_board()
        except Exception as e:
            print(f"Board scan failed: {e}")
        
        if board.turn == chess.WHITE:  # Human's turn
            print("Your turn (WHITE). Enter your move (e.g., e2e4) or command:")
            print("'undo' - undo last move")
            print("'edit' - edit piece positions") 
            print("'resign' - resign the game")
            
            user_input = input("Your move: ").strip().lower()
            
            if user_input == 'undo': # not a physical undo
                if len(board.move_stack) > 0:
                    board.pop()
                    save_board_state(board)
                    print("Last move undone.")
                else:
                    print("No moves to undo.")
                continue
            elif user_input == 'edit':
                print("Please manually adjust pieces and press Enter when done.")
                input("Press Enter to continue...")
                continue
            elif user_input == 'resign':
                print("You resigned. Game over.")
                break
            elif validate_human_move(board, user_input):
                move = chess.Move.from_uci(user_input)
                
                # Execute the move with Dobot

                from_sq = move.from_square
                to_sq = move.to_square
                from_pos = (chess.square_file(from_sq), 7 - chess.square_rank(from_sq))
                to_pos = (chess.square_file(to_sq), 7 - chess.square_rank(to_sq))
                
                print(from_sq, to_sq, from_pos, to_pos)
                
                exec_move(from_pos, to_pos, 
                        isCapture=board.is_capture(move),
                        promotion=move.promotion)
                
                board.push(move)
                save_board_state(board)
            else:
                print("Invalid move. Try again.")
                continue
        else:  # Robot's turn (BLACK)
            print("Robot's turn (BLACK). Thinking...")
            result = engine.play(board, chess.engine.Limit(time=2.0))
            move = result.move
            print(f"Robot plays: {move.uci()}")
            
            # Execute the move with Dobot
            from_sq = move.from_square
            to_sq = move.to_square
            from_pos = (chess.square_file(from_sq), 7 - chess.square_rank(from_sq))
            to_pos = (chess.square_file(to_sq), 7 - chess.square_rank(to_sq))
            
            exec_move(from_pos, to_pos, 
                    isCapture=board.is_capture(move),
                    promotion=move.promotion)
            
            board.push(move)
            save_board_state(board)
    
    print("Game over. Result:", board.result())
    engine.quit()
    cap.release()

if __name__ == "__main__":
    try:
        move_to_position(home_position[0], home_position[1], home_position[2] + 15)
        game_loop()
    except KeyboardInterrupt:
        print("\nGame interrupted by user")
    finally:
        move_to_position(home_position[0], home_position[1], home_position[2])
        engine.quit()
        cap.release()
        dobot.close()
