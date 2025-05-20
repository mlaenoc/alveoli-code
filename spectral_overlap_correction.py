import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

subdirectories = [""]

def process_image_set(dir, time, pos, offs):
    col3_file = os.path.join(dir, f"time{time}_pos{pos}_col3_offs{offs}_exp2.tiff")
    col4_file = os.path.join(dir, f"time{time}_pos{pos}_col4_offs{offs}_exp2.tiff")

    if not os.path.exists(col3_file) or not os.path.exists(col4_file):
        print(f"Files missing for time={time}, pos={pos}, offs={offs}, exp2")
        return

    try:
        image_A = cv2.imread(col3_file, cv2.IMREAD_ANYDEPTH).astype(np.uint16)
        image_B = cv2.imread(col4_file, cv2.IMREAD_ANYDEPTH).astype(np.uint16)
    except Exception as e:
        print(f"Error reading images: {e}")
        return
    scaling_factor = 1.2
    image_A_scaled = (image_A / scaling_factor).astype(np.uint16)
    subtracted_image = cv2.subtract(image_B,image_A_scaled)

    restored_image = subtracted_image
    output_path = os.path.join(dir, f"time{time}_pos{pos}_col6_offs{offs}_exp2.tiff")
    cv2.imwrite(output_path, restored_image.astype(np.uint16))
    print(f"Processed and saved: {output_path}")

for dir in subdirectories:
    print(f"Processing directory: {dir}")
    for time in range(1, 50):
        for pos in range(1, 51):
            for offs in range(1, 4):
            #    for exp in range(1, 3):
                    print(f"Processing time={time}, pos={pos}, offs={offs}, exp=2")
                    process_image_set(dir, time, pos, offs)
