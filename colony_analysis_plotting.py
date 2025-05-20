import pandas as pd
import matplotlib.pyplot as plt
import os
pxsize=0.00043
# Path to the total results file
dir_path = r''  # Ensure this is a raw string
excel_file = os.path.join(dir_path, 'results.xlsx')

# Define the worksheets to plot (indexing from 0)
sheets_to_plot = {1: 'grey', 3: 'red', 5: 'green', 6: 'blue'}  # 2nd, 3rd, 5th, and 6th sheets

# Load the Excel file
xls = pd.ExcelFile(excel_file)

# Create figure
plt.figure(figsize=(10, 6))

# Loop through selected worksheets
for i, (sheet_index, color) in enumerate(sheets_to_plot.items()):
    sheet_name = xls.sheet_names[sheet_index]  # Get the actual sheet name
    df = pd.read_excel(excel_file, sheet_name=sheet_name)

    # Extract Mean and StdDev columns
    mean_values = df["Mean"]
    std_values = df["StdDev"]

    # Plot Mean
    plt.plot(df.index * pxsize, mean_values, color=color, label=sheet_name, alpha=0.7)

    # Fill between Mean ± StdDev
    plt.fill_between(df.index * pxsize, mean_values - std_values, mean_values + std_values,
                     color=color, alpha=0.3)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(0,3)
plt.ylim(0,7.5)
plt.xlabel('Radius (mm)', fontsize=14)
plt.ylabel('Intensity (A.U.)', fontsize=14)
plt.gca().spines['top'].set_linewidth(1.5)
plt.gca().spines['right'].set_linewidth(1.5)
plt.gca().spines['left'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)
x_position24 = 0 # Set x position for the vertical line
x_error24 = 0 # Set the error range
x_position0=0
x_error0=0
# Draw a black vertical line at x_position
plt.axvline(x=x_position24, color='black', linestyle='-', linewidth=1.5, alpha=0.8)
# Fill the error region around the vertical line
plt.fill_betweenx(y=[plt.ylim()[0], plt.ylim()[1]],  # Fill from bottom to top of plot
                  x1=x_position24 - x_error24,
                  x2=x_position24 + x_error24,
                  color='black',
                  alpha=0.2)
plt.axvline(x=x_position0, color='black', linestyle='--', linewidth=1.5, alpha=0.8) # Fill the error region around the vertical line
plt.fill_betweenx(y=[plt.ylim()[0], plt.ylim()[1]],  # Fill from bottom to top of plot
                  x1=x_position0 - x_error0,
                  x2=x_position0 + x_error0,
                  color='black',
                  alpha=0.2)
plot_path = os.path.join(dir_path, '')
plt.savefig(plot_path, dpi=600)
plt.show()
