#!/usr/bin/python
#point_density_functions.py

import numpy as np
import pandas as pd
from scipy import stats
from laspy.file import File
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import plotly.graph_objects as go
import plotly.express as px

def raw_to_df(raw,column_names):
    '''function takes raw output of laspy.File.get_points() and column names, and returns a pandas Dataframe'''
    raw_list = [a[0].tolist() for a in raw]
    df = pd.DataFrame(raw_list,columns = column_names)
    return df

def scale_and_offset(df,header,append_to_df=False):
    '''Function takes as input the dataframe output of raw_to_df and the laspy header file.
       Output is a nx3 dataframe with adjusted X,Y, and Z coordinates, from the formula: 
       X_adj = X*X_scale + X_offset.
       Brooklyn LiDAR readings appear to be in feet, and use NAVD 88 in the vertical and 
       New York Long Island State Plane Coordinate System NAD 33 in the horizontal.'''
    offset = header.offset
    scale = header.scale
    scaled_xyz = df[['X','Y','Z']]*scale + offset
    if append_to_df:
        df['x_scaled'] = scaled_xyz['X']
        df['y_scaled'] = scaled_xyz['Y']
        df['z_scaled'] = scaled_xyz['Z'] 
        return df
    else:
        return scaled_xyz

def create_df_pickle(file_dir,filename,column_names):
    inFile = File(file_dir+filename, mode='r')
    raw = inFile.get_points()
    df = raw_to_df(raw,column_names)
    df = scale_and_offset(df,inFile.header,append_to_df=True)
    pickle_name = 'las_points_'+filename[34:40]+'.pkl'
    df.to_pickle(file_dir + pickle_name)

def create_df_hd5(file_dir,filename,column_names):
    inFile = File(file_dir+filename, mode='r')
    raw = inFile.get_points()
    df = raw_to_df(raw,column_names)
    df = scale_and_offset(df,inFile.header,append_to_df=True)
    hdf_name = 'las_points_'+filename[9:-4]+'.lz'
    df.to_hdf(file_dir + hdf_name,key='df',complevel=1,complib='lzo')
#    return df

def label_returns(las_df):
    '''
    Parses the flag_byte into number of returns and return number, adds these fields to las_df.
    Input - las_df - dataframe from .laz or .lz file
    Output - first_return_df - only the first return points from las_df.
           - las_df - input dataframe with num_returns and return_num fields added 
    '''
    
    las_df['num_returns'] = np.floor(las_df['flag_byte']/16).astype(int)
    las_df['return_num'] = las_df['flag_byte']%16
    first_return_df = las_df[las_df['return_num']==1]
    first_return_df = first_return_df.reset_index(drop=True)
    return first_return_df, las_df

# Load pickle, extract points around square, iterate
def grab_points(pt_files,file_dir,pt_x,pt_y,feet_from_point):
    '''
    Function extracts all points in all pt_files within feet_from_point of (pt_x,pt_y)
    Inputs:
        pt_files - List of strings, filenames of .lz files (created by create_df_hd5 function)
        file_dir - String, directory name containing pt_files
        pt_x,pt_y - Float, X and Y coordinate of the center point of the desired output
        feet_from_point - Float, how many feet in each coordinate direction to allow
        
    Output:
        square_points - DataFrame, contains only points within the bounds described, full LAS fields
    '''
    size_of_square = (2*feet_from_point)**2
    square_points = pd.DataFrame()
    for pick in pt_files:
        las_points = pd.read_hdf(file_dir+pick)
        if 'flight_id' not in las_points.columns:
            las_points['flight_id'] = pick[11:-3]
        new_square_points = las_points[ (las_points['x_scaled'] < pt_x + feet_from_point)
                &(las_points['x_scaled'] > pt_x - feet_from_point) 
                &(las_points['y_scaled'] < pt_y + feet_from_point)
                &(las_points['y_scaled'] > pt_y - feet_from_point)
              ]
        print("Point count in new square from {:s}: {:d}".format(pick,new_square_points.shape[0]))
        #pts_from_scan.append((pick,new_square_points.shape[0]))
        square_points = square_points.append(new_square_points,sort=True)

    print("Total point count in square: {:d}".format(square_points.shape[0]))
    print("Size of square: {:2.2f} sq ft".format(size_of_square))
    print("Point density: {:2.2f} points / sq ft".format(square_points.shape[0]/size_of_square))
    return square_points

def grab_points_big_rect(pt_files,file_dir,uv_inv,w):
    '''
    Function extracts all points in all pt_files within feet_from_point of (pt_x,pt_y)
    Note: This function currently only works in 2-D (horizontal plane)
    Inputs:
        pt_files - List of strings, filenames of .lz files (created by create_df_hd5 function)
        file_dir - String, directory name containing pt_files
        uv_inv - 
        w - 
        
    Output:
        square_points - DataFrame, contains only points within the bounds described, full LAS fields
    '''
    rectangle_points = pd.DataFrame()
    for pick in pt_files:
        las_points = pd.read_hdf(file_dir+pick)
        if 'flight_id' not in las_points.columns:
            las_points['flight_id'] = pick[11:-3]
        unit_square = (las_points[['x_scaled','y_scaled']]-w)@(uv_inv.T)
        new_rectangle_points = las_points[(unit_square[0]<=1) & (unit_square[0]>=0) & (unit_square[1]<=1) & (unit_square[1]>=0)]
        print("Point count in new square from {:s}: {:d}".format(pick,new_rectangle_points.shape[0]))
        #pts_from_scan.append((pick,new_square_points.shape[0]))
        rectangle_points = rectangle_points.append(new_rectangle_points,sort=True)

    print("Total point count in square: {:d}".format(rectangle_points.shape[0]))
    return rectangle_points



def plane_fit(square_points):
    '''
    Fits a plane via SVD to the provided points.
    Input: 
        (n x 3+) dataframe with fields x_scaled, y_scaled, and z_scaled
    Output: 
        normal vector - normal vector to plane fitted via MLS (3x1 numpy array)
        points - provided x,y,z points with zero mean (n x 3 numpy array)
        square_points - returns the dataframe with 'dist_from_plane' appended (n x 4+ dataframe)
        pts_on_plane - projection of x,y,z points onto the fitted plane (n x 3 numpy array)
    '''
    
    raw_points = np.array(square_points[['x_scaled','y_scaled','z_scaled']]).T
    points = raw_points.T - raw_points.mean(axis=1)
    svd = np.linalg.svd(points.T)
    norm_vector = np.transpose(svd[0])[2]    
    # Calculate each point's distance from the plane
    dist_from_plane = [np.dot(point,norm_vector) for point in points]

    # Project each point onto the plane
    proj_on_norm = dist_from_plane*np.array([norm_vector]).T
    pts_on_plane = points - proj_on_norm.T
    square_points.loc[:,'dist_from_plane'] = dist_from_plane
    
    return norm_vector,points,square_points,pts_on_plane

def prep_square_for_plotting(square_points,min_list=None):
    '''
    Function removes the min value in each coordinate from _plot fields, appends new fields
    Inputs:
        square_points - DataFrame with fields x_scaled, y_scaled, and z_scaled
        min_list (optional) - 3-tuple with (x_min,y_min,z_min) to be used.
    Output:
        square_points - Same DataFrame with x_plot,y_plot,z_plot fields added (_scaled field less the min value)
        min_list - [x_min,y_min,z_min] values used to create _plot fields
    '''
    square_points.loc[:,'size_num'] = 1 # added for plotting
    if min_list:
        x_min = min_list[0]
        y_min = min_list[1] 
        z_min = min_list[2]
    else:
        x_min = square_points['x_scaled'].min()
        y_min = square_points['y_scaled'].min()
        z_min = square_points['z_scaled'].min()
    square_points['x_plot'] = square_points['x_scaled'] - x_min
    square_points['y_plot'] = square_points['y_scaled'] - y_min
    square_points['z_plot'] = square_points['z_scaled'] - z_min
    min_list = [x_min,y_min,z_min]
    return square_points, min_list


def grab_wall_face(square_points,norm_vector,pt_1,z_low,z_high,epsilon=1e-2):
    '''
    Function extracts points from square_points dataframe that are in the x-y line defined by 2 points (+/-epsilon)
    and between [z_low,z_high] in the vertical.
    Inputs:
        square_points - dataframe with x_plot,y_plot,z_plot
        norm_vector - 3x1 numpy array of the normal vector to the plane
        pt_1 - 3-tuple of xyz coordinate of a point in the wall
        z_low,z_high - scalars indicating the range of vertical
        epsilon - scalar, indicating the allowable point distance from the plane defined by norm_vector and pt_1
    Output:
        wall_face - dataframe subset of square_points satisfying the above criteria
    '''
    xyz_array = square_points[['x_scaled','y_scaled','z_scaled']]
    dist_from_plane = norm_vector@(xyz_array - pt_1).T
    wall_face = square_points[(abs(dist_from_plane)<epsilon)&(square_points['z_scaled']>z_low) & (square_points['z_scaled']<z_high)]
    return wall_face

def in_horizontal_square(rectangle_points,center_point,feet_from_point):
    '''
    Function filters dataframe for points less than feet_from_point from center_point
    Note: this function only works in the xy plane
    
    Inputs:
    rectangle_points - (n x 3+) dataframe with fields x_scaled, y_scaled, and z_scaled
    center_point - 2x1 (x,y) point within rectangle
    feet_from_point - scalar
    
    Output:
    square_points - (n x 3+) filtered dataframe
    '''
    square_points = rectangle_points[ (rectangle_points['x_scaled'] < center_point[0] + feet_from_point)
            &(rectangle_points['x_scaled'] > center_point[0] - feet_from_point) 
            &(rectangle_points['y_scaled'] < center_point[1] + feet_from_point)
            &(rectangle_points['y_scaled'] > center_point[1] - feet_from_point)
          ]
    return square_points

def in_vertical_square(square_points,norm_vector,center_pt,horizontal_feet_from_pt, vertical_feet_from_pt):
    '''
    Function counts the number of points in the (2*horizontal_feet_from_pt)*(2*vertical_feet_from_pt) sqft vertical space.
    Allowed distance in the xy plane is determined by the normal vector to the planar normal vector.  Points are projected onto
    plane and then the xy dist is calculated.
    Inputs:
        square_points - DataFrame containing x_plot,y_plot,z_plot fields
        norm_vector - 3D numpy array of xyz directions, norm = 1
        center_pt - 3-tuple of xyz point, center of rectangle
        feet_from_pt - float, how many feet in each coordinate direction to allow
    Output:
        vertical_square - DataFrame containing the points within the rectangle, same format as square_points input
        density - scalar point density per sq ft for the rectangle
    '''
    xy_vector = np.array([-norm_vector[1],norm_vector[0]])
    xy_vector = xy_vector / np.linalg.norm(xy_vector) # Norm to 1


    # Project each point onto the plane
    orth_component = [dist*norm_vector for dist in square_points['dist_from_plane']]
    proj_on_plane = square_points[['x_scaled','y_scaled','z_scaled']] - np.array(orth_component)

    xy_dist_from_center_pt = np.linalg.norm(proj_on_plane[['x_scaled','y_scaled']] - center_pt[:2],axis=1)
    
    vertical_square = square_points[(xy_dist_from_center_pt<horizontal_feet_from_pt)
                                        &(square_points['z_scaled']<center_pt[2]+vertical_feet_from_pt) &
                                        (square_points['z_scaled']>center_pt[2]-vertical_feet_from_pt)]


    rect_area = ((vertical_feet_from_pt*2)*(horizontal_feet_from_pt*2))
    density = vertical_square.shape[0]/rect_area
    #print("Vertical density: {:2.3f} pts/sqft".format(density))
    #print("Rectangle area: {} sqft".format(rect_area))
    return vertical_square,density




### CLASSES FOR STATISTICAL SAMPLING

class FlightPath(object):
    '''
    FlightPath is an object that stores data about a single flight path for a given sample square.
    
    Attributes:
    flight_id - integer identifier of the flight (typically 0-40)
    norm_vector - 3x1 numpy array of the xyz coordinates of the norm vector
    avg_dist_from_plane - scalar average distance from fitted plane (fitted over all flight paths)
        for all points in the flight path
    
    '''
    def __init__(self,flight_id,norm_vector,avg_dist_from_plane=None):
        self.flight_id = flight_id
        self.norm_vector = norm_vector
        self.avg_dist_from_plane = avg_dist_from_plane
    
    def sd_dist_from_plane(self,square_points):
        self.sd_dist = np.std(square_points['dist_from_plane'])

class SampleSquare(object):
    '''
    Object to store all statistics for one sample square.
    
    Attributes:
    x,y - scalar coordinates of center point defining the square
    feet_from_point - scalar 1/2 length of one side of square
    nyc_/laefer_ flight_list - list of FlightPath objects from nyc or laefer dataset
    delta_h_matrix - k x k numpy array, where k is the number of flight paths and the entries are the difference
        in avg. point distance from the plane fit to the total point cloud 
    cosine_sim_matrix = k x k numpy array, where k is the number of flight paths and the entries are 
    cosine similarities between their normal vectors.
    '''
    def __init__(self, flight_list_laefer, flight_list_nyc=None,flight_list_usgs=None, x=None, y=None, feet_from_point=None):
        self.x = x
        self.y = y
        self.feet_from_point = feet_from_point
        self.flight_list_laefer = flight_list_laefer
        self.delta_h_matrix_laefer = self.delta_h_internal(self.flight_list_laefer)
        self.cosine_sim_matrix_laefer = self.cosine_sim_internal(self.flight_list_laefer)
        self.phi_laefer_total = self.phi_internal(self.flight_list_laefer,sample=False)
        self.phi_laefer_sample = self.phi_internal(self.flight_list_laefer,sample=True)
        if flight_list_nyc:
            self.flight_list_nyc = flight_list_nyc
            self.delta_h_matrix_nyc = self.delta_h_internal(self.flight_list_nyc)
            self.cosine_sim_matrix_nyc = self.cosine_sim_internal(self.flight_list_nyc)   
            self.phi_nyc_total = self.phi_internal(self.flight_list_nyc,sample=False)
            self.phi_nyc_sample = self.phi_internal(self.flight_list_nyc,sample=True)
        if flight_list_usgs:
            self.flight_list_usgs = flight_list_usgs
            self.delta_h_matrix_usgs = self.delta_h_internal(self.flight_list_usgs)
            self.cosine_sim_matrix_usgs = self.cosine_sim_internal(self.flight_list_usgs)   
            self.phi_usgs_total = self.phi_internal(self.flight_list_usgs,sample=False)
            self.phi_usgs_sample = self.phi_internal(self.flight_list_usgs,sample=True)

        
    def delta_h_internal(self,flight_list):
        # Calculates difference in avg dist from plane for all flight path pairs, returns a matrix
        delta_h_matrix = np.zeros((len(flight_list)-2,len(flight_list)-2))
        for i,flight_i in enumerate(flight_list[2:]): # Skip total and sampled
            for j,flight_j in enumerate(flight_list[2:]): # Skip total and sampled
                delta_h_matrix[i,j] = abs(flight_i.avg_dist_from_plane - flight_j.avg_dist_from_plane)
        return delta_h_matrix

    def cosine_sim_internal(self,flight_list):
        # Calculates cosine similarity for normal vectors of all flight path pairs, returns a matrix
        cosine_sim_matrix = np.zeros((len(flight_list)-2,len(flight_list)-2))
        for i,flight_i in enumerate(flight_list[2:]): # Skip total and sampled
            for j,flight_j in enumerate(flight_list[2:]): # Skip total and sampled
                cosine_sim_matrix[i,j] = flight_i.norm_vector @ flight_j.norm_vector
        return cosine_sim_matrix
    
    def phi_internal(self,flight_list,sample=False):
        avg_flight_paths = np.mean([flight.sd_dist for flight in flight_list[2:]])
        if sample:
            phi = flight_list[1].sd_dist / avg_flight_paths
        else:
            phi = flight_list[0].sd_dist / avg_flight_paths
        return phi


### FUNCTIONS FOR STATISTICAL SAMPLING

def center_point_sample(num_points,
                        bottom_left_pt,top_left_pt,bottom_right_pt=None,
                        u_length=800,v_length=-80,
                        border=[0.05,0.05]):
    '''
    Function returns random samples (num_points of them) within the rectangle defined by the points and lengths.
    
    Inputs:
    bottom_left_pt - 3x1 numpy array with xyz coordinate 
    top_left_pt - 3x1 numpy array with xyz coordinate 
    bottom_right_pt - 3x1 numpy array with xyz coordinate
    u_length - length in the u direction (bottom_left_pt -> top_left_pt)
    v_length - length in the u direction (bottom_left_pt -> bottom_right_pt)
    border - 2x1 list [border_u,border_v], portion of unit square on each edge to avoid
    
    Output:
    num_points x 2 numpy array of (x,y) values
    '''  
    
    unit_u = (top_left_pt - bottom_left_pt)/np.linalg.norm(top_left_pt-bottom_left_pt)
    unit_v = (bottom_right_pt - bottom_left_pt)/np.linalg.norm(bottom_right_pt - bottom_left_pt)
    u = unit_u*u_length
    v = unit_v*v_length
    uv = np.array([u,v]).T # Why the transpose?
            
    w = bottom_left_pt
    # Select random point on unit square within border
    border = np.array(border)
    square_side = np.array([1 - (2*border[0]),1 - (2*border[1])])
    st = border.reshape(2,1) + square_side.reshape(2,1)*np.random.rand(2,num_points)
    return (uv @ st + w.reshape((3,1))).T

def create_flight_list(square_points):
    '''
    flight_list creates a list of FlightPath objects, 
    1 for each unique flight_id plus 2 more (total and total_sampled).
    FlightPath object contains flight_id, norm_vector, std deviation of point distance from fitted plane.
    flight_id = -100: Full dataset
    flight_id = -200: Full dataset, sampled to the avg number of points in a single flight path
    
    Input:
    square_points: Dataframe for the square-around-a-point, including x_scaled,y_scaled, and z_scaled
    
    Output:
    flight_list (described above)
    '''
    
    # fit a plane for each flight
    flight_list = []

    # Full dataset
    norm_vector,_,square_points_total,_ = plane_fit(square_points)
    flightpath = FlightPath(-100,norm_vector)
    flightpath.sd_dist_from_plane(square_points_total)
    flight_list.append(flightpath)

    # Full dataset, sampled down
    flight_count = len(square_points['flight_id'].unique())
    density = square_points.shape[0] / flight_count
    square_points_sampled = square_points.sample(n=int(density))

    norm_vector,_,square_points_sample,_ = plane_fit(square_points_sampled)
    flightpath = FlightPath(-200,norm_vector)
    flightpath.sd_dist_from_plane(square_points_sampled)
    flight_list.append(flightpath)


    for flight_id in square_points['flight_id'].unique():
        # Avg distance from total point cloud plane - filtering square_points_total
        avg_dist_from_plane = square_points_total[square_points_total['flight_id']==flight_id]['dist_from_plane'].mean()

        norm_vector,_,square_points_flight,_ = plane_fit( \
                                               square_points[square_points['flight_id']==flight_id])
        flightpath = FlightPath(flight_id,norm_vector,avg_dist_from_plane)
        flightpath.sd_dist_from_plane(square_points_flight)
        flight_list.append(flightpath)       
        
    return flight_list