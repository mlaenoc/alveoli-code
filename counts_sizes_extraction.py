import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import os
import re
import math
from scipy.spatial import distance
threshold_saturation = 65500  # Replace with your threshold value
saturated_pixel_count = 500
centroids_by_time = {}
rows=[]
blue_centroid_list=[]
red_centroid_list=[]
green_centroid_list=[]
grey_centroid_list=[]

# Define constants
num_positions = 50
num_values_per_position = 15
num_times = 24

def sort_nicely( l ):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )
def split_contour_recursive(contour, split_threshold):
    if len(contour) < 5:
        return None  # Or skip/continue based on your logic
    ellipse = cv2.fitEllipse(contour)
    ellipse = cv2.fitEllipse(contour)
    (x, y), (major_axis, minor_axis), angle = ellipse

    if major_axis - minor_axis > split_threshold:
        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        split_contour1 = []
        split_contour2 = []
        for point in contour:
            if point[0][0] < cx:
                split_contour1.append([point])
            else:
                split_contour2.append([point])

        if len(split_contour1) > 2:
            split_contour1 = split_contour_recursive(np.array(split_contour1))
        if len(split_contour2) > 2:
            split_contour2 = split_contour_recursive(np.array(split_contour2))

        return np.concatenate((split_contour1, split_contour2))
    else:
        return contour
times=[]
greys=[]
reds=[]
blues=[]
greens=[]
posss=[]
a = 402.4888
b = 4500
c = 0.0021
total_threshes=[]
total_reds=[]
total_blues=[]
total_greens=[]
image_path2 = ''

img_fluobaseline=Image.open(image_path2)
img_array_fluofov= np.array(img_fluobaseline, dtype=np.float32)/255

subdirectories=[r'']
for direct in subdirectories:
    df = pd.DataFrame()
    #rescue_line_heigth=0
    print(direct)
    images=os.listdir(direct)
    sort_nicely(images)
    means=[]
    directory=direct
    coorsdir=directory+""
    dir=coorsdir+""
    with open(dir, 'r') as file:
        # Iterate over each line in the file
        for line in file:
            # Process each line
            row_elements = line.strip().split()
            # Add the list of elements to the rows list
            rows.append(row_elements)
    x1=0
    pos_counter = 0
    for row in rows:
        filename = row[0]
        y1 = row[1]
        y2 = row[2]
        x1 = row[3]
        x2 = row[4]
        y3 = row[5]
        y4 = row[6]
        x3 = row[7]
        x4 = row[8]
        parts = filename.split('_')
        numbers = {}
        for part in parts:
            var_name = ''.join(filter(str.isalpha, part))  # Extract alphabetic characters
            var_value = ''.join(filter(str.isdigit, part))  # Extract numeric characters
            if var_name and var_value:  # Ensure both variable name and value are not empty
                numbers[var_name] = int(var_value)

        # Only process up to the first 3 positions
        pos_counter += 1  # Increment the counter
        for time in range(1, 2):
            blue_pertime=[]
            green_pertime=[]
            red_pertime=[]
            threshes_pertime=[]
            grey = []
            blue = []
            green = []
            red = []
            tim = []
            poss = []
            if 'time' in numbers and numbers['time'] == time and numbers['col'] == 1:
                x1, x2, y1, y2 = map(int, [x1, x2, y1, y2])
                x3, x4, y3, y4 = map(int, [x3, x4, y3, y4])
                centroids=[]
                count=1
                try:
                    imred=Image.open(directory+"\\time"+str(numbers['time'])+"_"+"pos"+str(numbers['pos'])+"_col2_offs2_exp1.tiff")
                    imred = np.array(imred, dtype=np.float32)
                    smollred=imred[y1:y2,x1:x2]
                    smoll2red=imred[y3:y4,x3:x4]
                    red_pertime.append([smollred,smoll2red])
                    if time==1:
                            min_contour_area_threshold=50
                            imgs = [smollred,smoll2red]
                            split_threshold = 10
                            count=1
                            for img in imgs:
                                total_area = 0
                                contour_areas = []  # List to store contour areas
                                centroids = []  # List to store centroids for dispersion calculation
                                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                                gray = img.astype('uint8')
                                inverted_image = cv2.bitwise_not(gray)
                                binary = cv2.adaptiveThreshold(inverted_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 4)
                                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                valid_components_image = np.zeros_like(binary)
                                    # Get image dimensions for edge proximity
                                height, width = binary.shape
                                for contour in contours:
                                    contour_area = cv2.contourArea(contour)
                                    if contour_area > min_contour_area_threshold:
                                        hull = cv2.convexHull(contour)
                                        contour_area = cv2.contourArea(contour)
                                        hull_area = cv2.contourArea(hull)
                                        if hull_area == 0:
                                            continue
                                        solidity = contour_area / hull_area
                                        if solidity > 0.4:
                                            split_contour = split_contour_recursive(contour, split_threshold)
                                            if split_contour is not None and len(split_contour) > 0:
                                                cv2.drawContours(valid_components_image, [split_contour], -1, 255, thickness=cv2.FILLED)
                                                contour_areas.append(cv2.contourArea(split_contour))

                                kernel = np.ones((2, 2), np.uint8)
                                valid_components_image = cv2.dilate(valid_components_image, kernel, iterations=1)
                                overlay_image_red = inverted_image.copy()
                                overlay_image_red = cv2.cvtColor(overlay_image_red, cv2.COLOR_GRAY2BGR)
                                overlay_image_red[valid_components_image == 255] = [0, 0, 255]  # Red color for the overlay
                                contours_valid, _ = cv2.findContours(valid_components_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                labels = []
                                if not contours_valid == ():
                                    for valid_contour in contours_valid:
                                        area = cv2.contourArea(valid_contour)
                                        perimeter = cv2.arcLength(valid_contour, True)
                                        x, y, width, height = cv2.boundingRect(valid_contour)
                                        if height < width:
                                            width, height = height, width
                                        M = cv2.moments(valid_contour)
                                        if M["m00"] != 0:
                                            centroid_x = int(M["m10"] / M["m00"])
                                            centroid_y = int(M["m01"] / M["m00"])
                                            centroids.append([centroid_x, centroid_y])  # Store centroid for later clumping calculation
                                        else:
                                            centroid_x, centroid_y = 0, 0
                                        labels.append({
                                            'Image': ("time" + str(numbers['time']) + "_" + "pos" + str(numbers['pos']) + "CA_"),
                                            'Length': height,
                                            'Width': width,
                                            'Area': area,
                                            'Perimeter': perimeter,
                                            'Centroid_X': centroid_x,
                                            'Centroid_Y': centroid_y
                                        })
                                        total_area += area
                                else:
                                    labels.append({
                                        'Image': None,
                                        'Length': None,
                                        'Width': None,
                                        'Area': None,
                                        'Perimeter': None,
                                        'Centroid_X': None,
                                        'Centroid_Y': None
                                    })

                                    # Compute pairwise distances for centroids
                                if len(centroids) > 1:
                                    centroid_distances = distance.pdist(np.array(centroids), metric='euclidean')
                                    avg_pairwise_distance = np.mean(centroid_distances)  # Measure of how spread out the objects are
                                    print(f"Average Pairwise Distance (Clumping Measure): {avg_pairwise_distance}")
                                    centroids_np = np.array(centroids)  # Convert list to NumPy array for easier calculations
                                    avg_centroid_x = np.mean(centroids_np[:, 0])  # Mean of all x-coordinates
                                    avg_centroid_y = np.mean(centroids_np[:, 1])
                                else:
                                    avg_pairwise_distance = None
                                    print("Not enough objects to compute clumping measure.")
                                    avg_centroid_x = 0  # Mean of all x-coordinates
                                    avg_centroid_y = 0
                                    # Convert labels to DataFrame and save to Excel
                                labels_df_CA = pd.DataFrame(labels)
                                df = pd.concat([df, labels_df_CA], axis=1)
                                # Assign labels_df to df starting from column 4
                                red_centroid=(avg_centroid_x,avg_centroid_y)
                                red_centroid_list.append(red_centroid)
                    total_reds.append(red_pertime)
                except FileNotFoundError:
                    continue
    #END OF RED
                try:
                    imgreen = Image.open(directory + "\\time" + str(numbers['time']) + "_" + "pos" + str(numbers['pos']) + "_col3_offs1_exp1.tiff")
                    imgreen2 = Image.open(directory + "\\time" + str(numbers['time']) + "_" + "pos" + str(numbers['pos']) + "_col3_offs3_exp1.tiff")
                    imgreen = np.array(imgreen, dtype=np.float32)
                    imgreen2 = np.array(imgreen2, dtype=np.float32)
                    smoll3green = imgreen2[y1:y2, x1:x2]
                    smoll4green = imgreen2[y3:y4, x3:x4]
                    smollgreen = imgreen[y1:y2, x1:x2]
                    smoll2green = imgreen[y3:y4, x3:x4]
                    green_pertime.append([smollgreen,smoll2green,smoll3green,smoll4green])
                    count_smollgreen = np.sum(smollgreen > threshold_saturation)
                    count_smoll2green = np.sum(smoll2green > threshold_saturation)
                    count_smoll3green = np.sum(smoll3green > threshold_saturation)
                    count_smoll4green = np.sum(smoll4green > threshold_saturation)
                    if count_smollgreen > saturated_pixel_count or count_smoll2green > saturated_pixel_count or count_smoll3green > saturated_pixel_count or count_smoll4green > saturated_pixel_count:
                        imgreen = Image.open(directory + "\\time" + str(numbers['time']) + "_" + "pos" + str(numbers['pos']) + "_col3_offs1_exp2.tiff")
                        imgreen2 = Image.open(directory + "\\time" + str(numbers['time']) + "_" + "pos" + str(numbers['pos']) + "_col3_offs3_exp2.tiff")
                        imgreen = np.array(imgreen, dtype=np.float32)
                        smollgreen = imgreen[y1:y2, x1:x2]
                        smoll2green = imgreen[y3:y4, x3:x4]
                        imgreen2 = np.array(imgreen2, dtype=np.float32)
                        smoll3green = imgreen2[y1:y2, x1:x2]
                        smoll4green = imgreen2[y3:y4, x3:x4]
                    if time==1:
                            min_contour_area_threshold=20
                            imgs = [smollgreen,smoll2green,smoll3green,smoll4green]
                            split_threshold = 10
                            count=1
                            for img in imgs:
                                    total_area = 0
                                    contour_areas = []  # List to store contour areas
                                    centroids = []  # List to store centroids for dispersion calculation
                                    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                                    gray = img.astype('uint8')
                                    inverted_image = cv2.bitwise_not(gray)
                                    binary = cv2.adaptiveThreshold(inverted_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 17, 4)
                                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    valid_components_image = np.zeros_like(binary)
                                    # Get image dimensions for edge proximity
                                    height, width = binary.shape
                                    for contour in contours:
                                        contour_area = cv2.contourArea(contour)
                                        if contour_area > min_contour_area_threshold:
                                            hull = cv2.convexHull(contour)
                                            contour_area = cv2.contourArea(contour)
                                            hull_area = cv2.contourArea(hull)
                                            if hull_area == 0:
                                                continue
                                            solidity = contour_area / hull_area
                                            if solidity > 0.4:
                                                split_contour = split_contour_recursive(contour, split_threshold)
                                                if split_contour is not None and len(split_contour) > 0:
                                                    cv2.drawContours(valid_components_image, [split_contour], -1, 255, thickness=cv2.FILLED)
                                                    contour_areas.append(cv2.contourArea(split_contour))

                                    kernel = np.ones((2, 2), np.uint8)
                                    valid_components_image = cv2.dilate(valid_components_image, kernel, iterations=1)
                                    overlay_image_red = inverted_image.copy()
                                    overlay_image_red = cv2.cvtColor(overlay_image_red, cv2.COLOR_GRAY2BGR)
                                    overlay_image_red[valid_components_image == 255] = [0, 0, 255]  # Red color for the overlay
                                    valid_components_image_01 = valid_components_image // 255
                                    cutout = img * valid_components_image_01
                                    contours_valid, _ = cv2.findContours(valid_components_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    labels = []
                                    if not contours_valid == ():
                                        for valid_contour in contours_valid:
                                            area = cv2.contourArea(valid_contour)
                                            perimeter = cv2.arcLength(valid_contour, True)
                                            x, y, width, height = cv2.boundingRect(valid_contour)
                                            if height < width:
                                                width, height = height, width
                                            M = cv2.moments(valid_contour)
                                            if M["m00"] != 0:
                                                centroid_x = int(M["m10"] / M["m00"])
                                                centroid_y = int(M["m01"] / M["m00"])
                                                centroids.append([centroid_x, centroid_y])  # Store centroid for later clumping calculation
                                            else:
                                                centroid_x, centroid_y = 0, 0
                                            mask = np.zeros(cutout.shape, dtype=np.uint8)
                                            cv2.drawContours(mask, [valid_contour], -1, 255, thickness=cv2.FILLED)
                                            mean_val = cv2.mean(cutout, mask=mask)[0]
                                            if mean_val > 123.7:
                                                mean_val_high = mean_val
                                            else:
                                                mean_val_high = None
                                            num_splits = max(1, int(np.ceil(area / 150)))  # At least 1, split per 100 units
                                            for _ in range(num_splits):
                                                labels.append({
                                                'Image': ("time" + str(numbers['time']) + "_" + "pos" + str(numbers['pos']) + "SA"),
                                                'Length': height,
                                                'Width': width,
                                                'Area': area,
                                                'Perimeter': perimeter,
                                                'Centroid_X': centroid_x,
                                                'Centroid_Y': centroid_y
                                                })
                                            total_area += area
                                    else:
                                        labels.append({
                                            'Image': None,
                                            'Length': None,
                                            'Width': None,
                                            'Area': None,
                                            'Perimeter': None,
                                            'Centroid_X': None,
                                            'Centroid_Y': None
                                        })
                                        # Compute pairwise distances for centroids
                                    if len(centroids) > 1:
                                        centroid_distances = distance.pdist(np.array(centroids), metric='euclidean')
                                        avg_pairwise_distance = np.mean(centroid_distances)  # Measure of how spread out the objects are
                                        centroids_np = np.array(centroids)  # Convert list to NumPy array for easier calculations
                                        avg_centroid_x = np.mean(centroids_np[:, 0])  # Mean of all x-coordinates
                                        avg_centroid_y = np.mean(centroids_np[:, 1])
                                    else:
                                        avg_pairwise_distance = None
                                        print("Not enough objects to compute clumping measure.")
                                    # Convert labels to DataFrame and save to Excel
                                    labels_df_SA = pd.DataFrame(labels)
                                    df = pd.concat([df, labels_df_SA], axis=1)
                                    count += 1
                                    fluo_count = sum(1 for entry in labels if 'Fluo' in entry and entry['Fluo'] is not None)
                                    area_count = sum(1 for entry in labels if 'Area' in entry and entry['Area'] is not None)
                                    green_centroid=(avg_centroid_x,avg_centroid_y)
                                    green_centroid_list.append(green_centroid)
                    total_greens.append(green_pertime)
                except FileNotFoundError:
                    continue
    #END OF GREEN
                try:
                    imblue = Image.open(directory + "\\time" + str(numbers['time']) + "_" + "pos" + str(numbers['pos']) + "_col6_offs1_exp1.tiff")
                    imblue = np.array(imblue, dtype=np.float32)
                    smollblue = imblue[y1:y2, x1:x2]
                    smoll2blue = imblue[y3:y4, x3:x4]
                    count_smollblue = np.sum(smollblue > threshold_saturation)
                    count_smoll2blue = np.sum(smoll2blue > threshold_saturation)
                    imblue2 = Image.open(directory + "\\time" + str(numbers['time']) + "_" + "pos" + str(numbers['pos']) + "_col6_offs3_exp1.tiff")
                    imblue2 = np.array(imblue2, dtype=np.float32)
                    smoll3blue = imblue2[y1:y2, x1:x2]
                    smoll4blue = imblue2[y3:y4, x3:x4]
                    blue_pertime.append([smollblue,smoll2blue,smoll3blue,smoll4blue])
                    count_smoll3blue = np.sum(smoll3blue > threshold_saturation)
                    count_smoll4blue = np.sum(smoll4blue > threshold_saturation)
                    if count_smollblue > saturated_pixel_count or count_smoll2blue > saturated_pixel_count or count_smoll3blue > saturated_pixel_count or count_smoll4blue > saturated_pixel_count:
                        imblue = Image.open(directory + "\\time" + str(numbers['time']) + "_" + "pos" + str(numbers['pos']) + "_col6_offs1_exp2.tiff")
                        imblue = np.array(imblue, dtype=np.float32)
                        smollblue = imblue[y1:y2, x1:x2]
                        smoll2blue = imblue[y3:y4, x3:x4]
                        imblue2 = Image.open(directory + "\\time" + str(numbers['time']) + "_" + "pos" + str(numbers['pos']) + "_col6_offs3_exp2.tiff")
                        imblue2 = np.array(imblue2, dtype=np.float32)
                        smoll3blue = imblue2[y1:y2, x1:x2]
                        smoll4blue = imblue2[y3:y4, x3:x4]
                    if time==1:
                            min_contour_area_threshold=20
                            min_contour_area_threshold2=98
                            imgs = [smollblue,smoll2blue, smoll3blue,smoll4blue]
                            split_threshold = 10
                            count=1
                            for img in imgs:
                                total_area=0
                                contour_areas = []  # List to store contour areas
                                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                                gray = img.astype('uint8')  # Convert to grayscale
                                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                                # Processing with different images
                                inverted_image = cv2.bitwise_not(blurred)
                                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                                clahe_image = clahe.apply(blurred)
                                inverted_image2 = cv2.bitwise_not(clahe_image)
                                normalized_image = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
                                inverted_image3 = cv2.bitwise_not(normalized_image)
                                # Process first image
                                binary = cv2.adaptiveThreshold(inverted_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 4)
                                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                valid_components_image = np.zeros_like(inverted_image)
                                for contour in contours:
                                    contour_area = cv2.contourArea(contour)
                                    if contour_area > min_contour_area_threshold:
                                        hull = cv2.convexHull(contour)
                                        hull_area = cv2.contourArea(hull)
                                        if hull_area == 0:
                                            continue
                                        solidity = contour_area / hull_area
                                        if solidity > 0.4:
                                            split_contour = contour  # Replace with split_contour_recursive(contour, split_threshold) if needed
                                            cv2.drawContours(valid_components_image, [split_contour], -1, 255, thickness=cv2.FILLED)
                                            contour_areas.append(cv2.contourArea(split_contour))
                                # Process second image
                                binary = cv2.adaptiveThreshold(inverted_image2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 4)
                                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                valid_components_image2 = np.zeros_like(binary)
                                for contour in contours:
                                    contour_area = cv2.contourArea(contour)
                                    if contour_area > min_contour_area_threshold:
                                        hull = cv2.convexHull(contour)
                                        hull_area = cv2.contourArea(hull)
                                        if hull_area == 0:
                                            continue
                                        solidity = contour_area / hull_area
                                        if solidity > 0.4:
                                            split_contour = contour  # Replace with split_contour_recursive(contour, split_threshold) if needed
                                            cv2.drawContours(valid_components_image2, [split_contour], -1, 255, thickness=cv2.FILLED)
                                            contour_areas.append(cv2.contourArea(split_contour))
                                # Process third image
                                binary = cv2.adaptiveThreshold(inverted_image3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 4)
                                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                valid_components_image3 = np.zeros_like(binary)
                                for contour in contours:
                                    contour_area = cv2.contourArea(contour)
                                    if contour_area > min_contour_area_threshold:
                                        hull = cv2.convexHull(contour)
                                        hull_area = cv2.contourArea(hull)
                                        if hull_area == 0:
                                            continue
                                        solidity = contour_area / hull_area
                                        if solidity > 0.4:
                                            split_contour = contour  # Replace with split_contour_recursive(contour, split_threshold) if needed
                                            cv2.drawContours(valid_components_image3, [split_contour], -1, 255, thickness=cv2.FILLED)
                                            contour_areas.append(cv2.contourArea(split_contour))
                                # Combine results from all three processed images
                                result_image = np.zeros_like(inverted_image)
                                for y in range(inverted_image.shape[0]):
                                    for x in range(inverted_image.shape[1]):
                                        pixel1 = valid_components_image[y, x]
                                        pixel2 = valid_components_image2[y, x]
                                        pixel3 = valid_components_image3[y, x]
                                        if pixel1 == 255 or pixel2 == 255 or pixel3 == 255:
                                            result_image[y, x] = 255
                                        else:
                                            result_image[y, x] = pixel1
                                kernel = np.ones((2, 2), np.uint8)
                                result_image = cv2.erode(result_image, kernel, iterations=2)
                                overlay_image_red = inverted_image.copy()
                                overlay_image_red = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)
                                overlay_image_red[result_image == 255] = [0, 0, 255]  # Red color for the overlay
                                gray_image = cv2.cvtColor(overlay_image_red, cv2.COLOR_BGR2GRAY)
                                _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
                                contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                mask = np.zeros_like(binary_image)
                                for contour in contours:
                                    area = cv2.contourArea(contour)
                                    if area > min_contour_area_threshold:
                                        hull = cv2.convexHull(contour)
                                        hull_area = cv2.contourArea(hull)
                                        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
                                        n_parts = math.ceil(area / min_contour_area_threshold2)
                                        x, y, w, h = cv2.boundingRect(contour)
                                        aspect_ratio = w / h
                                        for i in range(1, n_parts):
                                            if aspect_ratio > 1:
                                                split_x = x + int(i * w / n_parts)
                                                cv2.line(mask, (split_x, y), (split_x, y + h), (0, 0, 0), 1)
                                            else:
                                                split_y = y + int(i * h / n_parts)
                                                cv2.line(binary_image, (x, split_y), (x + w, split_y), (0, 0, 0), 1)

                                overlay_image_red = cv2.bitwise_and(binary_image, mask)
                                valid_components_image_01 = overlay_image_red // 255
                                cutout = img * valid_components_image_01
                                contours_valid, _ = cv2.findContours(overlay_image_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                labels = []
                                if contours_valid:
                                    for valid_contour in contours_valid:
                                        area = cv2.contourArea(valid_contour)
                                        perimeter = cv2.arcLength(valid_contour, True)
                                        x, y, width, height = cv2.boundingRect(valid_contour)
                                        if height < width:
                                            width, height = height, width
                                        M = cv2.moments(valid_contour)
                                        if M["m00"] != 0:
                                            centroid_x = int(M["m10"] / M["m00"])
                                            centroid_y = int(M["m01"] / M["m00"])
                                            centroids.append([centroid_x, centroid_y])  # Store centroid for later clumping calculation
                                        else:
                                            centroid_x, centroid_y = 0, 0
                                        mask = np.zeros(cutout.shape, dtype=np.uint8)
                                        cv2.drawContours(mask, [valid_contour], -1, 255, thickness=cv2.FILLED)
                                        mean_val = cv2.mean(cutout, mask=mask)[0]
                                        mean_val_high = mean_val if mean_val > 41.4 else None
                                        num_splits = max(1, int(np.ceil(area / 100)))  # At least 1, split per 100 units
                                        for _ in range(num_splits):
                                            labels.append({
                                            'Image': ("time" + str(numbers['time']) + "_" + "pos" + str(numbers['pos']) + "PA"),
                                            'Length': height,
                                            'Width': width,
                                            'Area': area,
                                            'Perimeter': perimeter,
                                            #'Fluo': mean_val_high,
                                            'Centroid_X': centroid_x,
                                            'Centroid_Y': centroid_y
                                            })
                                        total_area += area
                                else:
                                    labels.append({
                                        'Image': None,
                                        'Length': None,
                                        'Width': None,
                                        'Area': None,
                                        'Perimeter': None,
                                        'Centroid_X': None,
                                        'Centroid_Y': None
                                    })

                                    # Compute pairwise distances for centroids
                                if len(centroids) > 1:
                                    centroid_distances = distance.pdist(np.array(centroids), metric='euclidean')
                                    avg_pairwise_distance = np.mean(centroid_distances)  # Measure of how spread out the objects are
                                    print(f"Average Pairwise Distance (Clumping Measure): {avg_pairwise_distance}")
                                    centroids_np = np.array(centroids)  # Convert list to NumPy array for easier calculations
                                    avg_centroid_x = np.mean(centroids_np[:, 0])  # Mean of all x-coordinates
                                    avg_centroid_y = np.mean(centroids_np[:, 1])
                                else:
                                    avg_pairwise_distance = None
                                    print("Not enough objects to compute clumping measure.")
                                    # Convert labels to DataFrame and save to Excel
                                labels_df_PA = pd.DataFrame(labels)
                                df = pd.concat([df, labels_df_PA], axis=1)
                                count += 1
                                fluo_count = sum(1 for entry in labels if 'Fluo' in entry and entry['Fluo'] is not None)
                                area_count = sum(1 for entry in labels if 'Area' in entry and entry['Area'] is not None)
                                blue_centroid=(avg_centroid_x,avg_centroid_y)
                                blue_centroid_list.append(blue_centroid)
                    total_blues.append(blue_pertime)
                except FileNotFoundError:
                        continue
        #END OF BLUE
df.to_excel(direct + "\\" + '_properties'+str(time)+'.xlsx', index=False)#, header=['ImageCA', 'LengthCA', 'WidthCA', 'AreaCA', 'PerimeterCA','Centroid_XCA', 'Centroid_YCA','ImageSA', 'LengthSA', 'WidthSA', 'AreaSA', 'PerimeterSA','Centroid_XSA', 'Centroid_YSA','ImagePA', 'LengthPA', 'WidthPA', 'AreaPA', 'PerimeterPA','Centroid_XPA', 'Centroid_YPA'])
