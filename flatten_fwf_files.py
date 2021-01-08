import numpy as np
import pandas as pd
import csv 
import argparse

# Necessary to import pypwaves_updated.py from parent directory
# import sys
# sys.path.append('../')
import pypwaves_updated as pw

parser = argparse.ArgumentParser()
parser.add_argument('--file_dir', required=True, help='Directory where .pls and .wvs file are')
parser.add_argument('--pls_file', required=True, help='should be a .pls file')
# parser.add_argument('--pulse_filename', required=True, help='output filename for pulse data')
# parser.add_argument('--wave_filename', requires=True, help='output filename for wave data')
opt = parser.parse_args()



# Load pulsewave object from file
pls_file = opt.file_dir + opt.pls_file
# "../../Data/fwf_data/F_150326_155833_T_315500_234000.pls"
pulsewave = pw.openPLS(pls_file)
flight = opt.pls_file.split('.')[0]
print("Flight: ",flight)

pulse_filename = flight+'_pulse_record.csv'
wave_filename = flight+'_waves.csv'

df = pd.DataFrame(columns = ['gps_timestamp', 
                                'offset_to_waves', 
                                'x_anchor', 
                                'y_anchor', 
                                'z_anchor', 
                                'x_target', 
                                'y_target', 
                                'z_target', 
                                'first_return', 
                                'last_return', 
                                'pulse_number', 
                                'pulse_descriptor', 
                                'reserved', 
                                'edge', 
                                'scan_direction', 
                                'facet', 
                                'intensity', 
                                'classification', 
                                'dx', 
                                'dy', 
                                'dz'])



## Write PulseRecord file
# opening the csv file in 'w' mode 
pulse_file = open(pulse_filename, 'w', newline ='') 

# Create header
pr = pulsewave.get_pulse(5)
pr_dict = pr.table_to_dict()

with pulse_file: 
    # identifying header   
    header = pr_dict.keys()
    writer = csv.DictWriter(pulse_file, fieldnames = header) 
      
    writer.writeheader() 
    for pulse_num in range(pulsewave.num_pulses):
        pr = pulsewave.get_pulse(pulse_num)
        pr_dict = pr.table_to_dict()
        writer.writerow(pr_dict) 
pulse_file.close()

# Write the wave file
wave_file = open(wave_filename, 'w', newline ='') 
with wave_file: 
    writer = csv.writer(wave_file)
    header = ['PulseNumber','SamplingNumber','SegmentNumber','Samples']
    writer.writerow(header)
    for pulse_num in range(pulsewave.num_pulses):
        pr = pulsewave.get_pulse(pulse_num)
        wv = pulsewave.get_waves(pr)
        for i in wv.segments:
            row = list(wv.segments[i][-1])
            if i == 2:
                row.insert(0,1)
            else:
                row.insert(0,0)
            row.insert(0,i)
            row.insert(0,pulse_num)
            writer.writerow(row)
wave_file.close()