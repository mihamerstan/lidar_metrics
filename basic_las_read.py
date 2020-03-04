import numpy as np
import pandas as pd
from laspy.file import File


# The columns are different if the .las file also has full waveform.
columns_point_cloud = [
    'X','Y','Z',
    'intensity',
    'flag_byte',
    'classification_flags',
    'classification_byte',
    'user_data',
    'scan_angle',
    'pt_src_id',
    'gps_time']

def raw_to_df(raw,column_names):
    '''function takes raw output of laspy.File.get_points() and column names, and returns a pandas Dataframe'''
    raw_list = [a[0].tolist() for a in raw]
    df = pd.DataFrame(raw_list,columns = column_names)
    return df

file_dir = "../Data/parking_lot/"
filename = "10552_NYU_M2 - Scanner 1 - 190511_172753_1 - originalpoints.laz"
def main():
	inFile = File(file_dir+filename, mode='r')
	raw = inFile.get_points()
	df = raw_to_df(raw,columns_point_cloud)

	df.to_csv("filename.txt")

if __name__ == "__main__":
    main()