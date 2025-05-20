import cv2
import matplotlib.pyplot as plt
import os
import re

def sort_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    l.sort(key=alphanum_key)

def adjust_box(imgrey, rect_coords1, output_file_path, filename):
    fig, ax = plt.subplots()

    # Draw the first red rectangle
    rect1 = plt.Rectangle((rect_coords1[0], rect_coords1[1]), rect_coords1[2] - rect_coords1[0], rect_coords1[3] - rect_coords1[1], linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect1)

    # Initialize coordinates for the second box
    x3 = rect_coords1[2] + 20
    x4 = x3 + (rect_coords1[2] - rect_coords1[0])
    rect_coords2 = [x3, rect_coords1[1], x4, rect_coords1[3]]

    # Draw the second blue rectangle
    rect2 = plt.Rectangle((rect_coords2[0], rect_coords2[1]), rect_coords2[2] - rect_coords2[0], rect_coords2[3] - rect_coords2[1], linewidth=2, edgecolor='b', facecolor='none')
    ax.add_patch(rect2)

    ax.imshow(imgrey)

    def on_key(event):
        # Adjust size and position based on key presses
        if event.key == 'shift':
            rect_coords1[2] = rect_coords1[0] + 1400
            rect_coords1[3] = rect_coords1[1] + 1400
        elif event.key == 'up':
            rect_coords1[1] -= 10
            rect_coords1[3] -= 10
        elif event.key == 'down':
            rect_coords1[1] += 10
            rect_coords1[3] += 10
        elif event.key == 'left':
            rect_coords1[0] -= 10
            rect_coords1[2] -= 10
        elif event.key == 'right':
            rect_coords1[0] += 10
            rect_coords1[2] += 10
        elif event.key == 'enter':
            print(f"First box new position: y1={rect_coords1[1]}, y2={rect_coords1[3]}, x1={rect_coords1[0]}, x2={rect_coords1[2]}")
            print(f"Second box new position: y1={rect_coords2[1]}, y2={rect_coords2[3]}, x1={rect_coords2[0]}, x2={rect_coords2[2]}")
            with open(output_file_path, "a") as f:
                f.write(f"{filename} {rect_coords1[1]} {rect_coords1[3]} {rect_coords1[0]} {rect_coords1[2]} {rect_coords2[1]} {rect_coords2[3]} {rect_coords2[0]} {rect_coords2[2]}\n")
            plt.close(fig)
            return

        # Update second box coordinates to mirror the first box
        rect_coords2[0] = rect_coords1[2] + 20
        rect_coords2[2] = rect_coords2[0] + (rect_coords1[2] - rect_coords1[0])
        rect_coords2[1] = rect_coords1[1]
        rect_coords2[3] = rect_coords1[3]

        # Update the positions of the rectangles
        rect1.set_xy((rect_coords1[0], rect_coords1[1]))
        rect1.set_width(rect_coords1[2] - rect_coords1[0])
        rect1.set_height(rect_coords1[3] - rect_coords1[1])
        rect2.set_xy((rect_coords2[0], rect_coords2[1]))
        rect2.set_width(rect_coords2[2] - rect_coords2[0])
        rect2.set_height(rect_coords2[3] - rect_coords2[1])
        fig.canvas.draw()

    def on_click(event):
        if event.button == 1:  # Left mouse button
            rect_coords1[0] = int(event.xdata)
            rect_coords1[1] = int(event.ydata)
            rect_coords1[2] = rect_coords1[0] + (rect_coords1[2] - rect_coords1[0])
            rect_coords1[3] = rect_coords1[1] + (rect_coords1[3] - rect_coords1[1])

            # Update second box coordinates accordingly
            rect_coords2[0] = rect_coords1[2] + 20
            rect_coords2[2] = rect_coords2[0] + (rect_coords1[2] - rect_coords1[0])
            rect_coords2[1] = rect_coords1[1]
            rect_coords2[3] = rect_coords1[3]

            rect1.set_xy((rect_coords1[0], rect_coords1[1]))
            rect2.set_xy((rect_coords2[0], rect_coords2[1]))
            fig.canvas.draw()

    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

subdirectories = ['']

for direct in subdirectories:
    print(direct)
    rows = []  # Clear the rows list for each directory
    images = os.listdir(direct)
    sort_nicely(images)
    directory = direct
    coorsdir = directory + ""
    coords_txt_path = coorsdir + ""

    output_file_path = os.path.join(coorsdir, "")

    with open(output_file_path, "w") as f:
        pass

    with open(coords_txt_path, 'r') as file:
        for line in file:
            row_elements = line.strip().split()
            rows.append(row_elements)

    for row in rows:
        filename = row[0]
        y1, y2, x1, x2, y3, y4, x3, x4 = map(int, row[1:])
        parts = filename.split('_')
        numbers = {}
        for part in parts:
            var_name = ''.join(filter(str.isalpha, part))
            var_value = ''.join(filter(str.isdigit, part))
            if var_name and var_value:
                numbers[var_name] = int(var_value)

        for time in range(1, 2):
            if 'time' in numbers and numbers['time'] == time and numbers['col'] == 1:
                try:
                    imgrey = cv2.imread(directory + "\\time" + str(numbers['time']) + "_pos" + str(numbers['pos']) + "_col1_offs1_exp1.tiff")
                    imgrey = cv2.cvtColor(imgrey*127, cv2.COLOR_BGR2RGB)

                    rect_coords1 = [x1, y1, x2, y2]

                    adjust_box(imgrey, rect_coords1, output_file_path, filename)

                except Exception as e:
                    print("Error:", e)
                    continue
