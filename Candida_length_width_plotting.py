import os
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

base_dir = ""  # Replace with the path to your top-level directory
lengths = []
widths = []

for root, _, files in os.walk(base_dir):
    if "properties.xlsx" in files:
        file_path = os.path.join(root, "properties.xlsx")
        print(f"Processing file in: {root}")
        df = pd.read_excel(file_path, mangle_dupe_cols=True)
        for i, col in enumerate(df.columns):
            if 'length' in col.lower() or 'width' in col.lower():
                if i > 0:
                    prev_col_value = str(df.iloc[:, i-1].iloc[1]).upper()
                    if 'CA' in prev_col_value:
                        if 'length' in col.lower():
                            length = df[col].dropna().values.tolist()
                            lengths.append(length)
                            next_col = df.columns[i+1]
                            width=df[next_col].dropna().values.tolist()
                            widths.append(width)
lengths = [item for sublist in lengths for item in sublist]
widths = [item for sublist in widths for item in sublist]

plt.figure(figsize=(9,6))
plt.hist(np.array(lengths) * 0.119, bins=200, color='tomato', alpha=0.7, label='Length')
plt.hist(np.array(widths) * 0.119, bins=200, color='firebrick', alpha=0.7, linewidth=0.8, label='Width')
plt.xlim(0, 20)
plt.xlabel('Size ($\mu$m)', fontsize=24)
plt.ylabel('Frequency', fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.gca().spines['top'].set_linewidth(1.5)
plt.gca().spines['right'].set_linewidth(1.5)
plt.gca().spines['left'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)
plt.legend(fontsize=20, frameon=False)
plt.tight_layout()
df = pd.DataFrame({'Length': lengths, 'Width': widths})
output_path = base_dir+"sizes.xlsx"
df.to_excel(output_path, index=False)
plt.savefig(base_dir+"CA_sizes_new.tiff", dpi=600)
plt.show()
