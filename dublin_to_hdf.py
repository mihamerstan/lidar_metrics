#!/usr/bin/python
#point_density_functions.py

import numpy as np
import pandas as pd
from laspy.file import File

# Dublin tile
file_dir = '../../Data/dublin_sample/'
filename = 'T_316000_234000.laz'

# Corresponds to LAS 1.2 Point Data Record Format 1
columns_dublin_pt_cloud = [
    'X',
    'Y',
    'Z',
    'intensity',
    'return_number_byte',
    'classification_byte',
    'scan_angle',
    'user_data',
    'pt_src_id',
    'gps_time']
    
create_df_hd5(file_dir,filename,columns_dublin_pt_cloud)