import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import os
import re
import math
import tifffile
from scipy.spatial import distance
def rescale_image(image):
    # Rescale the image to the range [0, 1]
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(image)
def sort_nicely( l ):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )

reds=[]
blues=[]
greens=[]
total_reds=[]
total_blues=[]
total_greens=[]
image_path2 = ''

img_fluobaseline=Image.open(image_path2)
img_array_fluofov= np.array(img_fluobaseline, dtype=np.float32)/255

subdirectories=['']
for direct in subdirectories:
    print(direct)
    images=os.listdir(direct)
    sort_nicely(images)
    means=[]
    directory=direct
    coorsdir=directory+""
    dir=coorsdir+""
    rows=[]
    with open(dir, 'r') as file:
        # Iterate over each line in the file
        for line in file:
            # Process each line
            row_elements = line.strip().split()
            # Add the list of elements to the rows list
            rows.append(row_elements)
    for time in range(1, 25):
        try:
              blue_pertime=[]
              green_pertime=[]
              red_pertime=[]
              for pos in range(1,51):
                if time == 1:
                      row=rows[pos-1]
                if time > 1:
                      row=rows[(50*(time-1)-1)+pos]
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
                    numbers[var_name] = int(var_value)
                if time==1:
                    rowcount=pos
                if time>1:
                    rowcount=pos+(time*50)
                y1=rows[rowcount][1]
                y2=rows[rowcount][2]
                x1=rows[rowcount][3]
                x2=rows[rowcount][4]
                y3=rows[rowcount][5]
                y4=rows[rowcount][6]
                x3=rows[rowcount][7]
                x4=rows[rowcount][8]
                threshes_pertime=[]
                grey = []
                blue = []
                green = []
                red = []
                tim = []
                poss = []
                x1, x2, y1, y2 = map(int, [x1, x2, y1, y2])
                x3, x4, y3, y4 = map(int, [x3, x4, y3, y4])
                imred=Image.open(directory+"\\obj1\\time"+str(time)+"_"+"pos"+str(pos)+"_col2_offs2_exp1.tiff")
                imred = np.array(imred, dtype=np.float32)
                smollred=imred[y1:y2,x1:x2]
                smoll2red=imred[y3:y4,x3:x4]
                red_pertime.append([smollred,smoll2red])
                imgreen = Image.open(directory+"\\obj1\\time"+str(time)+"_"+"pos"+str(pos)+ "_col3_offs2_exp2.tiff")
                imgreen = np.array(imgreen, dtype=np.float32)
                smollgreen = imgreen[y1:y2, x1:x2]
                smoll2green = imgreen[y3:y4, x3:x4]
                green_pertime.append([smollgreen,smoll2green])#,smoll3green,smoll4green])
                imblue = Image.open(directory+"\\obj1\\time"+str(time)+"_"+"pos"+str(pos)+ "_col6_offs1_exp2.tiff")
                imblue = np.array(imblue, dtype=np.float32)
                smollblue = imblue[y1:y2, x1:x2]
                smoll2blue = imblue[y3:y4, x3:x4]
                blue_pertime.append([smollblue,smoll2blue])#,smoll3blue,smoll4blue])

              total_reds=red_pertime
              total_greens=green_pertime
              total_blues=blue_pertime
              total_reds = [item for sublist in total_reds for item in sublist]
              total_blues = [item for sublist in total_blues for item in sublist]
              total_greens = [item for sublist in total_greens for item in sublist]
              all_images_combined_red = np.concatenate([img.flatten() for img in total_reds])
              all_images_combined_blue= np.concatenate([img.flatten() for img in total_blues])
              all_images_combined_green= np.concatenate([img.flatten() for img in total_greens])
            # Find the minimum and maximum values
              ar=np.min(all_images_combined_red)
              br=np.max(all_images_combined_red)
              ag=np.min(all_images_combined_green)
              bg=np.max(all_images_combined_green)
              ab=np.min(all_images_combined_blue)
              bb=np.max(all_images_combined_blue)
              rescaled_reds = total_reds
              rescaled_greens = total_greens
              rescaled_blues = total_blues
              summed_reds=np.sum(rescaled_reds, axis=0)
              summed_greens=np.sum(rescaled_greens, axis=0)
              summed_blues=np.sum(rescaled_blues, axis=0)
              alpha_reds=summed_reds/len(total_reds)
              alpha_greens=summed_greens/len(total_greens)
              alpha_blues=summed_blues/len(total_blues)
              red_image_rescaled = rescale_image(alpha_reds)
              green_image_rescaled = rescale_image(alpha_greens)
              blue_image_rescaled = rescale_image(alpha_blues)
                # Stack the three rescaled images to form an RGB image
              rgb_image = np.stack((red_image_rescaled, green_image_rescaled, blue_image_rescaled), axis=-1)
              rgb_image_8bit = (rgb_image * 255).astype(np.uint8)
        # Save as an 8-bit TIFF
              tifffile.imwrite(direct+"" + str(time) + ".tiff",
                     rgb_image_8bit,
                     photometric='rgb')  # Add this to ensure ImageJ recognizes it as RGB
        except FileNotFoundError:
            continue
