import cv2
import apriltag
import numpy as np
import json
import chess
import chess.engine
import time

import threading    

from pydobot import Dobot
from segment import segment_image

CAMERA_INDEX = 3



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
    
    # Convert board coordinates to robot coordinates
    from_x, from_y, from_z = board_to_robot(*from_pos)
    to_x, to_y, to_z = board_to_robot(*to_pos)
    
    z_low_main = z_axis_limits[0]  # Z position when lowered
    z_high_main = z_axis_limits[1]  # Z position when raised
    
    z_low_aux = z_axis_limits[2]
    z_high_aux = z_axis_limits[3]

    # Handle capture (move captured piece to auxiliary board first)
    if isCapture:
        # Get free slot in auxiliary board
        aux_x, aux_y, aux_z = get_aux_coord()
        
        # Move to capture position and pick up piece
        move_to_position(to_x, to_y, to_z + 10)
        move_to_position(to_x, to_y, to_z - 8)
        dobot.suck(True)
        time.sleep(suction_delay)
        move_to_position(to_x, to_y, to_z + 10)
        
        # Move to auxiliary board and drop piece
        move_to_position(aux_x, aux_y, aux_z + 10)
        move_to_position(aux_x, aux_y, aux_z - 8)
        dobot.suck(False)
        time.sleep(suction_delay)
        move_to_position(aux_x, aux_y, aux_z + 10)
    
    # Handle promotion
    if promotion:
        # Get promotion piece from auxiliary board
        aux_x, aux_y = get_aux_coord()
        
        # Move to auxiliary board and pick up promotion piece
        move_to_position(aux_x, aux_y, z_high_aux)
        move_to_position(aux_x, aux_y, z_low_aux)
        dobot.suck(True)
        time.sleep(suction_delay)
        move_to_position(aux_x, aux_y, z_high_aux)
        
        # Move to destination and drop promotion piece
        move_to_position(to_x, to_y, z_high_main)
        move_to_position(to_x, to_y, z_low_main)
        dobot.suck(False)
        time.sleep(suction_delay)
        move_to_position(to_x, to_y, z_high_main)
        
        # Move original piece to auxiliary board
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
    move_to_position(from_x, from_y, from_z + 15)
    move_to_position(from_x, from_y, from_z)
    dobot.suck(True)
    time.sleep(suction_delay)
    move_to_position(from_x, from_y, from_z + 15)
    
    # Move to destination position and drop piece
    move_to_position(to_x, to_y, to_z + 15)
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
    tags = detect_apriltags(img)
    print(tags)
    if len(tags) < 4:
        raise Exception("Could not detect enough AprilTags for alignment")
    
    # Warp the perspective to get a clean board image
    board_markers = np.float32([
        [tags[0].corners[1][0] + 25, tags[0].corners[1][1] - 5],
        [tags[1].corners[0][0] - 25, tags[1].corners[0][1] - 5],
        [tags[2].corners[2][0] + 25, tags[2].corners[2][1] + 5],
        [tags[3].corners[3][0] - 25, tags[3].corners[3][1] + 5]
    ])
    
    board_size = 800
    dest_points = np.array([
        [board_size - 1, 0],
        [0, 0],
        [board_size - 1, board_size - 1],
        [0, board_size - 1]
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