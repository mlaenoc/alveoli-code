import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import os

directory = ""

data = {}  # Dictionary to store averages for each subfolder

for a in range(1, 43):
    dir2 = os.path.join(directory, str(a))
    ims = os.listdir(dir2)
    averages = []

    print(f"Processing folder {a}")

    for im in ims:
        imred = Image.open(os.path.join(dir2, im))
        imred = np.array(imred, dtype=np.float32)
        img = cv2.normalize(imred, None, 0, 255, cv2.NORM_MINMAX)
        gray = img.astype('uint8')
        inverted_image = cv2.bitwise_not(gray)
        binary = cv2.adaptiveThreshold(inverted_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 71, 4)
        _, otsu_binary = cv2.threshold(inverted_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_binary = cv2.bitwise_not(otsu_binary)
        intersection_binary = cv2.bitwise_and(binary, otsu_binary)

        intersection_binary_clean = cv2.medianBlur(intersection_binary, 5)
        intersection_binary_clean = cv2.medianBlur(intersection_binary_clean, 7)
        intersection_binary_clean = cv2.medianBlur(intersection_binary_clean, 5)

        masked_pixels = imred[intersection_binary_clean == 255]
        average_pixel_value = np.mean(masked_pixels) if masked_pixels.size > 0 else 0
        averages.append(average_pixel_value)

    data[f"Folder_{a}"] = averages

# Convert dictionary to DataFrame and save as Excel
df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))  # Convert uneven lists into a DataFrame
excel_path = os.path.join(directory, "averages.xlsx")
df.to_excel(excel_path, index=False)

print(f"Averages saved to {excel_path}")

# Plot the averages for visualization
plt.figure(figsize=(10, 6))
for key in data:
    plt.plot(data[key], label=key)
plt.ylim(0, 20000)
plt.legend()
plt.show()
