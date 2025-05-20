import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import re

# Folder containing the TIFF files
folder_path = r''

# Get the list of TIFF files in the folder
tiff_files = [f for f in os.listdir(folder_path) if f.endswith('.tiff')]

def sort_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    l.sort(key=alphanum_key)

sort_nicely(tiff_files)

# Iterate through each TIFF file
for tiff_file in tiff_files:
    file_path = os.path.join(folder_path, tiff_file)

    # Read the image as a 3-channel RGB
    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

    if image is not None and len(image.shape) == 3:
        # Split the image into R, G, B channels
        b_channel, g_channel, r_channel = cv2.split(image)

        # Compute the mean along the y-axis
        mean_red = np.mean(r_channel, axis=1) / np.max(np.mean(r_channel, axis=1))
        mean_green = np.mean(g_channel, axis=1) / np.max(np.mean(g_channel, axis=1))
        mean_blue = np.mean(b_channel, axis=1) / np.max(np.mean(b_channel, axis=1))

        std_red = np.std(r_channel, axis=1) / np.max(np.mean(r_channel, axis=1))
        std_green = np.std(g_channel, axis=1) / np.max(np.mean(g_channel, axis=1))
        std_blue = np.std(b_channel, axis=1) / np.max(np.mean(b_channel, axis=1))

        # Reverse y-axis order naturally
        ypos = np.linspace(len(mean_red) - 1, 0, len(mean_red)) * 0.119

        # Plot the results
        plt.figure(figsize=(9.34, 9.34))
        plt.plot(mean_blue, ypos, color='blue',  alpha=0.7)
        plt.plot(mean_green, ypos, color='green',  alpha=0.7)
        plt.plot(mean_red, ypos, color='red',  alpha=0.7)

        plt.plot(1000, 1000, color='blue', lw=10,label='PA', alpha=0.7)
        plt.plot(1000, 1000, color='green', lw=10,label='SA', alpha=0.7)
        plt.plot(1000, 1000, color='red', lw=10,label='CA', alpha=0.7)

        plt.fill_betweenx(ypos, mean_blue - std_blue, mean_blue + std_blue, color='blue', alpha=0.3)
        plt.fill_betweenx(ypos, mean_red - std_red, mean_red + std_red, color='red', alpha=0.3)
        plt.fill_betweenx(ypos, mean_green - std_green, mean_green + std_green, color='green', alpha=0.3)

        # Increase axis line thickness
        ax = plt.gca()
        ax.spines["top"].set_linewidth(1.5)
        ax.spines["right"].set_linewidth(1.5)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)

        # Set y-axis ticks (from 155 to 0)
        ax.set_xlim(0,1.5)
        ax.set_ylim(0,150)
        y_ticks = np.linspace(len(mean_red) - 1, 0, 6) * 0.119  # Reduce number of ticks
        plt.yticks(y_ticks, np.linspace(155, 0, 6).astype(int))  # Match tick labels

        ax.tick_params(axis="both", which="major", labelsize=36, width=3, length=10)

        # Labels
        plt.xlabel('Mean intensity (A.U.)', fontsize=40)
        plt.ylabel('y position ('+'$\mu$m)', fontsize=40)
        ax.legend(fontsize=24, frameon=False)
        plt.tight_layout()
        # Save the plot
        plot_file_path = os.path.join(folder_path, f'{os.path.splitext(tiff_file)[0]}_mean_plot_x.tiff')
        plt.savefig(plot_file_path, dpi=300)
        plt.close()
