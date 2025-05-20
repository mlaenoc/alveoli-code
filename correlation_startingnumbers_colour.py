import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d

#rowtime=3
def count_non_empty_cells(file_path):
    """Counts non-empty cells for specific grouped columns in an Excel file."""
    df = pd.read_excel(file_path, sheet_name=0)  # Read the first sheet

    # Define column groups
    group_PA1 = [43, 50]
    group_PA2 = [57, 64]

    group_SA1 = [15, 22]
    group_SA2 = [29, 36]

    group_CA1 = [1]
    group_CA2 = [8]


    num_blocks = 50  # Number of blocks
    block_size = 70  # Distance between block starts

    counts_PA = []
    counts_SA = []
    counts_CA = []

    for block in range(num_blocks):
        # Compute block-wise column indices
        pa1_cols = [col + block * block_size for col in group_PA1]
        pa2_cols = [col + block * block_size for col in group_PA2]
        sa1_cols = [col + block * block_size for col in group_SA1]
        sa2_cols = [col + block * block_size for col in group_SA2]
        ca1_cols = [col + block * block_size for col in group_CA1]
        ca2_cols = [col + block * block_size for col in group_CA2]

        # Ensure column indices are within bounds
        max_columns = df.shape[1]
        pa1_cols = [col for col in pa1_cols if col < max_columns]
        pa2_cols = [col for col in pa2_cols if col < max_columns]
        sa1_cols = [col for col in sa1_cols if col < max_columns]
        sa2_cols = [col for col in sa2_cols if col < max_columns]
        ca1_cols = [col for col in ca1_cols if col < max_columns]
        ca2_cols = [col for col in ca2_cols if col < max_columns]

        # Compute non-empty cell counts
        counts_PA.append(df.iloc[:, pa1_cols].notna().sum().sum() if pa1_cols else 0)
        counts_PA.append(df.iloc[:, pa2_cols].notna().sum().sum() if pa2_cols else 0)

        counts_SA.append(df.iloc[:, sa1_cols].notna().sum().sum() if sa1_cols else 0)
        counts_SA.append(df.iloc[:, sa2_cols].notna().sum().sum() if sa2_cols else 0)

        counts_CA.append(df.iloc[:, ca1_cols].notna().sum().sum() if ca1_cols else 0)
        counts_CA.append(df.iloc[:, ca2_cols].notna().sum().sum() if ca2_cols else 0)

    return counts_PA, counts_SA, counts_CA

for rowtime in range(15,16):
    all_counts_PA = []
    all_counts_SA = []
    all_counts_CA = []
    # Loop through directories and process files
    dirs=[""]
    CAQ1=[]
    CAQ2=[]
    CAQ3=[]
    CAQ4=[]
    PAQ1=[]
    PAQ2=[]
    PAQ3=[]
    PAQ4=[]
    SAQ1=[]
    SAQ2=[]
    SAQ3=[]
    SAQ4=[]

    for base_dir in dirs:
        for root, bit, files in os.walk(base_dir+""):
            if "properties.xlsx" in files:
                file_path = os.path.join(root, "properties.xlsx")
                pa_counts, sa_counts, ca_counts = count_non_empty_cells(file_path)
                all_counts_PA.extend(pa_counts)
                all_counts_SA.extend(sa_counts)
                all_counts_CA.extend(ca_counts)
        if os.path.exists(base_dir+""):
                df=pd.read_excel(base_dir+"")
                if len(df.columns) >= 100:  # Ensure there are at least 102 columns (index 0 to 100)
                        averages = np.clip(df.iloc[rowtime, 0:101], 0, None).tolist()  # Rows 13-15 (0-based index: 12-14), Columns 2-100 (1-based index: 1-100)
                        CAQ1.append(averages)
        if os.path.exists(base_dir+""):
                df=pd.read_excel(base_dir+"")
                if len(df.columns) >= 100:  # Ensure there are at least 102 columns (index 0 to 100)
                        averages = np.clip(df.iloc[rowtime, 0:101], 0, None).tolist()  # Rows 13-15 (0-based index: 12-14), Columns 2-100 (1-based index: 1-100)
                        PAQ1.append(averages)
        if os.path.exists(base_dir+""):
                df=pd.read_excel(base_dir+"")
                if len(df.columns) >= 100:  # Ensure there are at least 102 columns (index 0 to 100)
                        averages = np.clip(df.iloc[rowtime, 0:101], 0, None).tolist()  # Rows 13-15 (0-based index: 12-14), Columns 2-100 (1-based index: 1-100)
                        SAQ1.append(averages)
    CAQ1=[item for sublist in CAQ1 for item in sublist]
    CAQ1 = np.array(CAQ1)
    PAQ1=[item for sublist in PAQ1 for item in sublist]
    PAQ1 = np.array(PAQ1)
    SAQ1=[item for sublist in SAQ1 for item in sublist]
    SAQ1 = np.array(SAQ1)
    ratioC_PS = [a / b if b != 0 else a for a, b in zip(all_counts_CA,all_counts_PA)]
    ratioPS_C = [a / b if b != 0 else a for a, b in zip(all_counts_PA,all_counts_CA)]
    def normalize_to_255(array):
        min_val = np.min(array)
        max_val = np.max(array)
        if max_val > min_val:  # Avoid division by zero
            return ((array - min_val) / (max_val - min_val) * 255).astype(int)
        else:
            return np.zeros_like(array, dtype=int)  # If all values are the same, set to 0
    # Normalize values to 0–255 for RGB coloring
    CAQ1_norm = normalize_to_255(CAQ1)
    SAQ1_norm = normalize_to_255(SAQ1)
    PAQ1_norm = normalize_to_255(PAQ1)
    colors = np.stack([CAQ1_norm, SAQ1_norm, PAQ1_norm], axis=1) / 255  # Normalize to 0–1 for matplotlib

def generate_binned_image(all_counts_CA, all_counts_SA, all_counts_PA, CAQ1, SAQ1, PAQ1, output_file='binned_image.png'):
    # Compute bin edges

    all_counts_CA=np.array(all_counts_CA)
    all_counts_PA=np.array(all_counts_PA)
    all_counts_SA=np.array(all_counts_SA)

    x_counts =  all_counts_PA+all_counts_SA
    print(x_counts)
    y_counts = all_counts_CA
    print(y_counts)
    #culo
    # Define bin edges for a 10x10 grid
    x_bins = np.linspace(x_counts.min(), x_counts.max(), 51)
    y_bins = np.linspace(y_counts.min(), y_counts.max(), 51)

    # Compute mean values for each color channel in the bins
    r_binned, _, _, _ = binned_statistic_2d(x_counts, y_counts, CAQ1, statistic='mean', bins=[x_bins, y_bins])
    g_binned, _, _, _ = binned_statistic_2d(x_counts, y_counts, SAQ1, statistic='mean', bins=[x_bins, y_bins])
    b_binned, _, _, _ = binned_statistic_2d(x_counts, y_counts, PAQ1, statistic='mean', bins=[x_bins, y_bins])

    # Normalize to [0, 255] (handling NaN values by setting them to zero)
    def normalize(arr):
        arr = np.nan_to_num(arr, nan=0)  # Replace NaNs with 0 (black bins)
        if arr.max() > 0:
            arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
        return arr.astype(np.uint8)

    r_img = normalize(r_binned).T  # Transpose to match image convention
    g_img = normalize(g_binned).T
    b_img = normalize(b_binned).T

    # Stack channels to create an RGB image
    img = np.stack([r_img, g_img, b_img], axis=-1)

    # Resize to 100x100 using nearest-neighbor interpolation
    img_large = np.kron(img, np.ones((50, 50, 1), dtype=np.uint8))

    # Save and show the image
    plt.imshow(img_large)
    plt.xlabel("PA+SA")
    plt.ylabel("CA")
    # Invert the y-axis
    plt.gca().invert_yaxis()

    # Update tick labels to show counts instead of pixel positions
    x_tick_positions = np.linspace(0, img_large.shape[1], len(x_bins))[::10]  # Every 2nd tick
    y_tick_positions = np.linspace(0, img_large.shape[0], len(y_bins))[::10]  # Every 2nd tick

    # Convert pixel positions to counts
    x_tick_labels = (x_tick_positions / 50) * (x_bins[1] - x_bins[0]) + x_bins[0]
    y_tick_labels = (y_tick_positions / 50) * (y_bins[1] - y_bins[0]) + y_bins[0]

    # Round values
    x_tick_labels = np.round(x_tick_labels).astype(int)
    y_tick_labels = np.round(y_tick_labels).astype(int)

    plt.xticks(x_tick_positions, x_tick_labels)
    plt.yticks(y_tick_positions, y_tick_labels)

    plt.show()
# Example usage (assuming CA_counts, SA_counts, PA_counts, CAq1, SAq1, and PAq1 are NumPy arrays)
generate_binned_image(all_counts_CA, all_counts_SA, all_counts_PA, CAQ1, SAQ1, PAQ1)
