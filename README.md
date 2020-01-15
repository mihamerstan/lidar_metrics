# lidar_fwf
Python3 tools for full waveform lidar processing.  Based on pypwaves (see below) but ported to python 3 and expanded to cover more of the Pulsewaves spec.

# Resources:
Pulsewaves specification: https://github.com/PulseWaves/Specification/blob/master/specification.rst
GeoTIFF specification (Pulsewaves utilizes the GeoKey spec): http://docs.opengeospatial.org/is/19-008r4/19-008r4.html#_requirements_class_geokeydirectorytag

pypwaves - only python library I've found for parsing pulsewaves format, but written in Python 2 (and doesn't address geokey records, which is what we're using)
https://github.com/adamchlus/pypwaves

Laspy - LAS file library, doesn't seem to support FWF issues.
Laspy documentation: https://pythonhosted.org/laspy/laspy_tools.html

Python structs documentation: https://docs.python.org/2/library/struct.html
