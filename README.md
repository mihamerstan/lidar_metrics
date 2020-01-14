# lidar_fwf
Tools for full waveform lidar processing.

Resources:
Pulsewaves specification: https://github.com/PulseWaves/Specification/blob/master/specification.rst

pypwaves - only python library I've found for parsing pulsewaves format, but written in Python 2 (and doesn't address geokey records, which is what we're using)
https://github.com/adamchlus/pypwaves

Laspy - LAS file library, doesn't seem to support FWF issues.
Laspy documentation: https://pythonhosted.org/laspy/laspy_tools.html
