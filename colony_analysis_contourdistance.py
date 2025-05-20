import cv2
import numpy as np
import os
import pandas as pd
from glob import glob

# Define the folder containing TIFF images
folder_path = r""  # Change this to your folder path

# Output Excel files
distance_excel = os.path.join(folder_path, "contour_distances.xlsx")
metrics_excel = os.path.join(folder_path, "contour_metrics.xlsx")

# Get all TIFF images in the folder
tif_files = glob(os.path.join(folder_path, "*.tif"))

# Find the first image that starts with "contour"
contour_image_path = next((f for f in tif_files if os.path.basename(f).startswith("contour")), None)

if not contour_image_path:
    print("No contour image found! Exiting.")
    exit()

# Load the contour image
contour_image = cv2.imread(contour_image_path, cv2.IMREAD_UNCHANGED)

if contour_image is None:
    print("Failed to load the contour image! Exiting.")
    exit()

# Convert to grayscale if the image has more than one channel
if len(contour_image.shape) > 2:
    contour_image = cv2.cvtColor(contour_image, cv2.COLOR_BGR2GRAY)

# Define the white contour pixel value
contour_value = 0
# Create a binary mask where contour pixels are 255, everything else is 0
binary_mask = np.where(contour_image == contour_value, 255, 0).astype(np.uint8)

# Find contours in the contour image
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if not contours:
    print("No contour found in the reference image! Exiting.")
    exit()

# Get the largest contour
contour = max(contours, key=cv2.contourArea)

# Compute centroid
M = cv2.moments(contour)
centroid_x = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
centroid_y = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0

print(f"Centroid found at: ({centroid_x}, {centroid_y})")

# Compute distances of each contour point from the centroid
distances = []
for point in contour:
    x, y = point[0]
    distance = np.sqrt((x - centroid_x) ** 2 + (y - centroid_y) ** 2)
    distances.append(distance)

# Save distances to an Excel file
df_distances = pd.DataFrame({"Distance": distances})
df_distances.to_excel(distance_excel, index=False)
print(f"Distances saved to: {distance_excel}")

# Compute contour metrics
area = cv2.contourArea(contour)
perimeter = cv2.arcLength(contour, closed=True)

# Save area and perimeter to a separate Excel file
df_metrics = pd.DataFrame({"Metric": ["Area", "Perimeter"], "Value": [area, perimeter]})
df_metrics.to_excel(metrics_excel, index=False)
print(f"Metrics saved to: {metrics_excel}")
