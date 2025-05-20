import cv2
import numpy as np
from PIL import Image
import os
import tifffile

def check_and_convert_to_rgb(img_data):
    """
    Check if the image is in RGB format (3 channels). If not, convert it to RGB.
    """
    # Check the number of channels in the image
    if img_data.ndim == 3 and img_data.shape[2] == 3:
        # Already in RGB format
        return img_data
    else:
        # Convert to RGB if it's not in RGB format
        # For example, convert from GBR to RGB if needed
        img_data_rgb = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
        return img_data_rgb

def sum_rgb_images(final_list):
    summed_images = []

    # Iterate through each position in the final list and sum images
    for image_list in final_list:
        summed_image = None

        # Iterate over all images at the current position
        for img in image_list:
            img_data = np.array(img, dtype=np.float32)  # Convert to float for summing
            img_data = check_and_convert_to_rgb(img_data)  # Ensure image is in RGB format

            if summed_image is None:
                summed_image = np.zeros_like(img_data, dtype=np.float32)

            # Sum the images
            summed_image += img_data

        # Perform the division by 7 and round to the nearest integer
        summed_image = summed_image / 7
        summed_image = np.round(summed_image).astype(np.uint16)  # Round and convert to uint16

        # Clip to ensure it stays within the 16-bit range (0-65535)
        summed_image = np.clip(summed_image, 0, 65535)

        # Append the summed image to the list
        summed_images.append(summed_image)

    return summed_images

# Function to sort files numerically
def sort_nicely(l):
    import re
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    l.sort(key=alphanum_key)

# Set up directories and initialize variables
subdirectories = [r'']

im_list = []
# First, load all images into sublists per directory
for direct in subdirectories:
    im_sublist = []
    images = os.listdir(direct)
    sort_nicely(images)  # Sort filenames nicely
    for im in images:
        if im.endswith("tiff") and im.startswith(""):
            ima = Image.open(os.path.join(direct, im))
            im_sublist.append(ima)  # Append each image to the sublist
    im_list.append(im_sublist)  # Append the sublist to im_list

# Determine the maximum number of images in any directory
max_images_per_position = max(len(sublist) for sublist in im_list)

# Create a new list where each position contains the images from the same index across directories
final_list = []

# Iterate over each image position
for pos in range(max_images_per_position):
    images_at_pos = []
    for sub in im_list:
        if pos < len(sub):
            images_at_pos.append(sub[pos])  # Append image if it exists at this position
    final_list.append(images_at_pos)

output_directory = r''

# Call the function on the final list
summed_images = sum_rgb_images(final_list)

# Iterate through the summed images and save each one as a TIFF file
for i, image in enumerate(summed_images):
    output_path = os.path.join(output_directory, f'summed_image_{i+1}.tiff')

    # Save the image in RGB format (uint16)
    tifffile.imwrite(output_path, image, photometric='rgb')
    print(f"Saved: {output_path}")
