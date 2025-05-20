from PIL import Image
import os
import numpy as np

# Folder containing the images
input_folder = r""
output_folder = r""
os.makedirs(output_folder, exist_ok=True)
# Image indices to stitch (1-based)
indices_to_stitch = [1 + 7 * i for i in range(15)]  # [1, 8, 15, ..., 99]

# Dimensions of the stitched image
rows, cols = 9, 5  # 5 rows, 3 columns

def create_positional_matrix():
    """
    Create the specified positional matrix:
    5,  6, 15
    4,  7, 14
    3,  8, 13
    2,  9, 12
    1, 10, 11
    """
    return [5, 6, 15, 4, 7, 14, 3, 8, 13, 2, 9, 12, 1, 10, 11]

def reorder_tiles(tiles, positional_matrix):
    """
    Reorder tiles according to the positional matrix.
    """
    reordered = [tiles[idx - 1] for idx in positional_matrix]  # Convert 1-based to 0-based indexing
    return reordered

def stitch_images(tiles, rows, cols):
    """
    Stitch images into a rectangle based on the given rows and columns.
    """
    tile_width, tile_height = tiles[0].size
    canvas = Image.new('I;16', (cols * tile_width, rows * tile_height))  # Create a 16-bit canvas

    for idx, tile in enumerate(tiles):
        row, col = divmod(idx, cols)
        x_offset = col * tile_width
        y_offset = row * tile_height
        canvas.paste(tile, (x_offset, y_offset))

    return canvas

# Load and process images
all_images = sorted(os.listdir(input_folder))
all_images = [img for img in all_images if img.lower().endswith('.tiff')]

# Ensure we only use valid indices
valid_indices_to_stitch = [i - 1 for i in indices_to_stitch if i - 1 < len(all_images)]  # Convert to 0-based

if len(valid_indices_to_stitch) < len(indices_to_stitch):
    print(f"Warning: Not all expected indices are available. Using {len(valid_indices_to_stitch)} instead of {len(indices_to_stitch)}.")

# Select images
selected_images = [all_images[i] for i in valid_indices_to_stitch]
tiles = [Image.open(os.path.join(input_folder, img)) for img in selected_images]

# Convert all tiles to 16-bit grayscale mode (if not already)
tiles = [tile.convert('I;16') for tile in tiles]

# Divide each original tile by 40
tiles = [Image.fromarray((np.array(tile, dtype=np.float32) / 40).astype(np.uint16)) for tile in tiles]

# Load and rescale the correction image
corrected = Image.open("").convert('I;16')
corrected_array = np.array(corrected, dtype=np.float32)

# Rescale the array to [0, 1]
corrected_min = corrected_array.min()
corrected_max = corrected_array.max()
rescaled_array = (corrected_array - corrected_min) / (corrected_max - corrected_min)

# Correct each tile by dividing by the rescaled correction image
corrected_tiles = []
for tile in tiles:
    tile_array = np.array(tile, dtype=np.float32)
    corrected_tile_array = tile_array / rescaled_array
    corrected_tile_array = np.clip(corrected_tile_array, 0, 65535)  # Ensure it remains in the 16-bit range
    corrected_tile = Image.fromarray(corrected_tile_array.astype(np.uint16))
    corrected_tiles.append(corrected_tile)

# Generate positional matrix and reorder tiles
positional_matrix = create_positional_matrix()
reordered_tiles = reorder_tiles(corrected_tiles, positional_matrix)

# Create the stitched image
stitched_image = stitch_images(reordered_tiles, rows, cols)

# Display the stitched image
stitched_image.show()

# Save the stitched image
output_path = os.path.join(output_folder, "stitched_image.tif")
stitched_image.save(output_path)
print(f"Saved: {output_path}")
