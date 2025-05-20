import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# File path
input_fileP = r''
input_fileS = r''
input_fileC = r''

dirs = [input_fileP, input_fileS, input_fileC]
output_file = r''

# Read the Excel file
colours = ["blue", "green", "red"]
labels = ["PA", "SA", "CA"]
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)

for dir, col, lab in zip(dirs, colours, labels):
    df = pd.read_excel(dir)

    # Extract columns
    time = df.iloc[:, 0]  # First column: Time
    fluorescence = df.iloc[:, 1]  # Second column: Fluorescence
    std_dev = df.iloc[:, 2]  # Third column: Standard Deviation

    # Compute upper and lower bounds for shading
    lower_bound = fluorescence - std_dev
    upper_bound = fluorescence + std_dev

    # Create the plot
    plt.plot(time, fluorescence, color=col, linewidth=2, label=lab)
    plt.fill_between(time, lower_bound, upper_bound, color=col, alpha=0.3)

# Formatting
plt.xlabel('Time (h)', fontsize=20)
plt.ylabel('Fluorescence (A.U.)', fontsize=20)
plt.ylim(0, 65535)
plt.xlim(0, 12)

# Customize axis appearance
for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(1.5)
    ax.tick_params(direction='out', length=8, width=2)
    ax.tick_params(direction='out', length=6, width=2, which="minor")

# Keep top and right spines but remove their ticks
for axis in ['top', 'right']:
    ax.spines[axis].set_linewidth(1.5)
ax.tick_params(top=False, right=False)  # Remove ticks on top and right axes

plt.tick_params(axis='both', which='major', labelsize=15)
plt.legend(fontsize=20, frameon=False)
# Save and show the plot
fig.tight_layout()
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.show()
