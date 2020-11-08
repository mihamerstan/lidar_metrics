# lidar_fwf
Tooling used for the paper [Metrics for Aerial, Urban LiDAR Point Clouds](https://arxiv.org/abs/2010.09951)
Python3 tools for full waveform lidar processing.  Based on pypwaves (see below) but ported to python 3 and expanded to cover more of the Pulsewaves spec.

# Resources:
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
