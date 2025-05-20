import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from PIL import Image
import tifffile

def sort_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    l.sort(key=alphanum_key)
kymograph=[]
prekymograph=[]
subdirectories = [r""]

for direct in subdirectories:
    print("Processing directory:", direct)
    images = os.listdir(direct)
    sort_nicely(images)
    for time in range (1,21):
        im = np.array(Image.open(os.path.join(direct, f"summed_image_{time}.tiff")))
                # Average across the x-axis
        avg = np.mean(im, axis=1, keepdims=True)
        strip = np.repeat(avg, 70, axis=1)
                # Append to the kymographs
        prekymograph.append(strip)
    kymograph = np.concatenate(prekymograph, axis=1)
    output_directory = direct
    output_filename = "kymograph.tif"
    output_filepath = os.path.join(output_directory, output_filename)
    tifffile.imwrite(output_filepath, kymograph.astype(np.uint8))
