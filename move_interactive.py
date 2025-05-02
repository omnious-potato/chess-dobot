import time
from pydobot import Dobot
import serial
import readchar  # for simple key detection (no enter needed)

STEP = 15.0
Z_STEP = 2.0
# Connect to Dobot

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

device = try_connecting_dobot()


def move_relative(dx=0, dy=0, dz=0):
    pose = device.pose()
    x, y, z, r = pose[:4]  # only take the first four values
    new_x = x + dx
    new_y = y + dy
    new_z = z + dz
    print(f"Moving to X:{new_x:.1f}, Y:{new_y:.1f}, Z:{new_z:.1f}")
    device.move_to(new_x, new_y, new_z, r, wait=True)


print("Control Dobot with keys:")
print("W/S = Y axis forward/back")
print("A/D = X axis left/right")
print("Q/E = Z axis up/down")
print("ESC = Quit")

try:
    while True:
        key = readchar.readkey()
        if key == 'w':
            move_relative(dy=STEP)
        elif key == 's':
            move_relative(dy=-STEP)
        elif key == 'a':
            move_relative(dx=-STEP)
        elif key == 'd':
            move_relative(dx=STEP)
        elif key == 'q':
            move_relative(dz=Z_STEP)
        elif key == 'e':
            move_relative(dz=-Z_STEP)
        elif key == "=":
            STEP += 1.0
            print(f"Step is set to {STEP}")
        elif key == "-":
            if STEP > 1.0:
                STEP -= 1.0
            
            print(f"Step is set to {STEP}")
        elif key == "]":
            pose = device.pose()

            with open("manual_markers.txt", "a") as file:
                file.write(f"{pose[0]} {pose[1]} {pose[2]}\n")
                print(f"Written position X:{pose[0]}, Y:{pose[1]}, Z:{pose[2]}")

        elif key == readchar.key.ESC:
            print("Exiting...")
            break
        else:
            print("Unknown key:", key)
except KeyboardInterrupt:
    print("Stopped by user.")
finally:
    device.close()
