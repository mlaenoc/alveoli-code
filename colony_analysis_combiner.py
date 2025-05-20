import os
import pandas as pd
from glob import glob

# Define base directory
base_dir = r''

# Find all subfolders
subfolders = [f.path for f in os.scandir(base_dir) if f.is_dir()]

# Storage for processed data
combined_data = {f"Sheet{i+1}": [] for i in range(7)}

# Iterate through subfolders to find the required files
for subfolder in subfolders:
    bg_file = os.path.join(subfolder, 'background_intensity.xlsx')
    radial_file = os.path.join(subfolder, 'radial_profiles.xlsx')

    # Check if both files exist
    if os.path.exists(bg_file) and os.path.exists(radial_file):
        print(f"Processing: {subfolder}")

        # Read background intensity values (assuming they are in the second column)
        bg_df = pd.read_excel(bg_file)
        bg_values = bg_df.iloc[:, 1].values  # Get values from the second column

        # Read the radial profiles file
        xls = pd.ExcelFile(radial_file)

        # Process each of the 7 worksheets
        for i, sheet_name in enumerate(xls.sheet_names[:7]):  # Ensure only 7 sheets are processed
            df = pd.read_excel(radial_file, sheet_name=sheet_name)

            # Normalize by background value
            df_normalized = df / bg_values[i]

            # Append to the combined dataset
            combined_data[f"Sheet{i+1}"].append(df_normalized)

# Create final Excel writer
output_file = os.path.join(base_dir, 'total_results.xlsx')
with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    for sheet_name, sheet_data in combined_data.items():
        # Concatenate all dataframes in this sheet
        if sheet_data:
            combined_df = pd.concat(sheet_data, axis=1)  # Combine along columns

            # Compute row-wise mean and std deviation
            combined_df["Mean"] = combined_df.mean(axis=1)
            combined_df["StdDev"] = combined_df.std(axis=1)

            # Save to the final Excel file
            combined_df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Total Excel file saved at: {output_file}")
