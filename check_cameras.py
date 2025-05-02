import cv2

def capture_from_all_devices(max_devices=10):
    for index in range(max_devices):
        cap = cv2.VideoCapture(index)

        if cap.isOpened():
            print(f"Device {index} opened successfully.")
            ret, frame = cap.read()
            if ret:
                filename = f"{index}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved image from device {index} to {filename}")
            else:
                print(f"Failed to read frame from device {index}")
            cap.release()
        else:
            print(f"Device {index} could not be opened.")

if __name__ == "__main__":
    capture_from_all_devices(max_devices=10)
