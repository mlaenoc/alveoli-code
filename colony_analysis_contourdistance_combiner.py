import os
import pandas as pd
from glob import glob

# Define the base directory
base_dir = r""
output_file = os.path.join(base_dir, "combined_contour_distances.xlsx")

# Find all subfolders
subfolders = [f.path for f in os.scandir(base_dir) if f.is_dir()]

# List to store all distances
all_distances = []

# Iterate over subfolders
for subfolder in subfolders:
    file_path = os.path.join(subfolder, "contour_distances.xlsx")

    # Check if the file exists
    if os.path.exists(file_path):
        df = pd.read_excel(file_path, usecols=[0])  # Read only the first column
        distances = df.iloc[:, 0].dropna().tolist()  # Convert to list, drop NaNs
        all_distances.extend(distances)

# Save the combined data to an Excel file
output_df = pd.DataFrame({"Distance": all_distances})
output_df.to_excel(output_file, index=False)

print(f"Combined data saved to {output_file}")
