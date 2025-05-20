# alveoli-code
Supporting code

Scripts used to analyze data and generate plots for the publication (insert DOI)

- squares_checker.py is a python script that allows the visualization and interactive adjustment of 2 square regions of interest in an image. It can take a set of coordinates that need adjsuting/checking.
- spectral_overlap_correction.py takes 2 fluorescence images (A and B) with a one-way spectral overlap (A is partially visible in B) and removes the signal overlapping signal (removes A from B).
- colony_stitching_with_correction.py reads a set of images and stitches them into a single one based on given overlaps and a matrix of positions. Each image is normalized by the illumination profile of the light source.
- colony_analysis_combiner.py reads the radial intensity profiles of multiple images stored in different excel files and combines them into 1 file.
- colony_analysis_contourdistance.py calculates contour distance from centroid, contour perimeter and area.
- colony_analysis_contourdistance_combiner.py combines all of the distances of contours from their centroids across multiple files in different directories.
- colony_analysis_plotting.py produces plots in Fig. 1.
- correlation_startingnumbers_colours.py produces a plot that correlates abundances of different species stored in an excel file with colour in an image.
- counts_sizes_extraction.py segments images from different fluorescent channels and extracts length, width, area, perimeter and position of the objects.
- merger_differentalveoli_pre_x_average.py extracts square regions of interest from different images and merges them.
- merger_differentexperiments_pre_x_average.py merges images from different experiments.
- mothermachine_fluorescence_Pseudomonas.py performs a total segmentation of the fluorescent regions of interest in the image and extracts the mean fluorescence value.
- mothermachine_fluorescence_Candida.py performs a total segmentation of the fluorescent regions of interest in the image and extracts the mean fluorescence value.
- mothermachine_fluorescence_Staph.py performs a total segmentation of the fluorescent regions of interest in the image and extracts the mean fluorescence value.
- mothermachine_fluorescence_plotting.py plots fluorescence profiles over time for 3 different time series.
- x_average_fluorescence_profile.py extracts the average fluorescence profile n
- x_average_kymograph.py
