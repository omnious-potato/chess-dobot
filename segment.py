import time
import os
import cv2

from pylibdmtx.pylibdmtx import decode
from multiprocessing import Pool

def decode_cell(img):
    if img is None:
        return None
    return decode(img, max_count=1, timeout=50, shrink=2)

BOARD_IMG_PATH = "./board_warped.jpg"
THREADS = 4
def segment_image(img=None, cellsX=8, cellsY=8, cellSize=100, debug=False):
    if img is None:
        img = cv2.imread(BOARD_IMG_PATH)
    
    if img is None:
        return None
    
    cell_images =[]
    for x in range(cellsX):
        for y in range(cellsY):
            Y = y * cellSize
            X = x * cellSize
            cell = img[Y:(Y+cellSize),X:(X+cellSize)]
            cell_images.append(cell)
    end = time.time()

    if debug:
        for i, x in enumerate(cell_images):
            cv2.imwrite(f"cell{i}.png",x)
    
    start = time.time()
    with Pool(processes=THREADS) as pool:
        results = pool.map(decode_cell, cell_images)
    end = time.time()

    # print(f"Processing with {THREADS} threads took {end - start}s")
    os.remove(BOARD_IMG_PATH)

    # print(results)
    return results