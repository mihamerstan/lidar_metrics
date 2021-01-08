# lidar_fwf
Tooling used for the paper [Metrics for Aerial, Urban LiDAR Point Clouds](https://arxiv.org/abs/2010.09951).  

# File Descriptions
 * [pypwaves_updated.py](https://github.com/mihamerstan/lidar_fwf/blob/main/pypwaves_updated.py): pypwaves is a python library for parsing the pulsewaves full waveform LiDAR format, but it is incomplete and written for python2. This file updates pypwaves for python3 and fills out more of the pulsewaves spec.
 * [point_density_functions.py](https://github.com/mihamerstan/lidar_fwf/blob/main/point_density_function.py): This file contains the majority of sampling and statistical functionality utilized in the paper.
 * [flatten_fwf_files.py](https://github.com/mihamerstan/lidar_fwf/blob/main/flatten_fwf_files.py): Utilizes pypwaves_updated to flatten pulsewave .pls+.wvs files into .csv files. Not utilized in the paper. 
 
# Notebooks
 * [Sampling_and_Analysis_SunsetPark.ipynb](https://github.com/mihamerstan/lidar_fwf/blob/master/notebooks/Sampling_and_Analysis_SunsetPark.ipynb): This notebook performs the central sampling and analysis of the paper. For each sample surface (defined by a set of xyz coordinates on the plane), the notebook generates the desired number of sample squares, collects the points in that square, generates the desired statistics, and aggregates over all the sample surfaces. 
 * [Sampling_and_Analysis_Dublin.ipynb](https://github.com/mihamerstan/lidar_fwf/blob/master/notebooks/Sampling_and_Analysis_Dublin.ipynb): Same as previous, but for Dublin dataset.
 * [trig_section.ipynb](https://github.com/mihamerstan/lidar_fwf/blob/master/notebooks/trig_section.ipynb): Support for a lot of the Methodology section of the paper (e.g., angle of capture impact on point density)
 * [fig12_mean_ortho_offset.ipynb](https://github.com/mihamerstan/lidar_fwf/blob/master/notebooks/fig12_mean_ortho_offset.ipynb): Generates figure 12 in the paper.
 * [num_flights_nyc.ipynb](https://github.com/mihamerstan/lidar_fwf/blob/master/notebooks/num_flights_nyc.ipynb): This notebook identifies and appends flight IDs for the NYC 2017 dataset based on time gaps between points.
 * [num_flights_usgs.ipynb](https://github.com/mihamerstan/lidar_fwf/blob/master/notebooks/num_flights_usgs.ipynb): This notebook identifies and appends flight IDs for the USGS dataset based on time gaps between points.
 * [projected_overall_density.ipynb](https://github.com/mihamerstan/lidar_fwf/blob/master/notebooks/projected_overall_density.ipynb): Calculates projected density for a given LAS file.

# Other Python LiDAR Resources:
Pulsewaves specification: https://github.com/PulseWaves/Specification/blob/master/specification.rst
GeoTIFF specification (Pulsewaves utilizes the GeoKey spec): http://docs.opengeospatial.org/is/19-008r4/19-008r4.html#_requirements_class_geokeydirectorytag

pypwaves - only python library I've found for parsing pulsewaves format, but written in Python 2 (and doesn't address geokey records, which is what we're using)
https://github.com/adamchlus/pypwaves

Laspy - LAS file library, doesn't seem to support FWF issues.
Laspy documentation: https://pythonhosted.org/laspy/laspy_tools.html

Installing laszip: https://stackoverflow.com/questions/49500149/laspy-cannot-find-laszip-when-is-installed-from-source-laszip-is-in-path

Geo-references for Brooklyn LiDAR run:
NAD83: https://spatialreference.org/ref/epsg/nad83-new-york-long-island-ftus/
NAVD 88: https://en.wikipedia.org/wiki/North_American_Vertical_Datum_of_1988
