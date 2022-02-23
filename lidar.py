import math
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d import Axes3D
import time
from pr2_utils import read_data_from_csv,lidar_to_vehicle_transformation, tic,toc,bresenham2D
from trajectory import vehicle_to_world_transformation,get_vehicle_pos



##------  Initialise Rotation Matrices  ------##



##--------  FOG  --------##
# FOG sensor measurements are stored as [timestamp, delta roll, delta pitch, delta yaw] in radians.

##--------  STEREO  --------##
# Baseline of stereo cameras: 475.143600050775 (mm)

##--------  LIDAR  --------##
# LiDAR rays with value 0.0 represent infinite range observations.
# FOV: 190 (degree)
# Start angle: -5 (degree)
# End angle: 185 (degree)
# Angular resolution: 0.666 (degree)
# Max range: 80 (meter)

##--------  Get data from CSV files  --------##

# Encoder Data

def run_lidar():
    filename = 'sensor_data/lidar.csv'
    timestamp,data = read_data_from_csv(filename)

    timestamp = (timestamp-timestamp[0])/1000000000

    lidar_3d_in_vehicle_frame = lidar_to_vehicle_transformation(data)   # Function which transforms Lidar values from lidar frame to vehicle frame 

    lidar_3d_in_world_frame = vehicle_to_world_transformation(lidar_3d_in_vehicle_frame,timestamp)

 
    # for i in range(0,lidar_3d_in_world_frame.shape[0]):
    #     for j in range(0,lidar_3d_in_world_frame.shape[1]):
    #         if(np.logical_and(lidar_3d_in_world_frame[i,j,0]>8,lidar_3d_in_world_frame[i,j,1]>8)):
    #             lidar_3d_in_world_frame[i,j,0]=0
    #             lidar_3d_in_world_frame[i,j,1]=0

    # x_c,y_c = np.loadtxt('trajectory.txt')      # later on, call this using trajectory.py
    # plt.scatter(lidar_3d_in_world_frame[:,:,0],lidar_3d_in_world_frame[:,:,1],s=1,color = "m")

    # plt.plot(x_c,y_c,color="r")
    # plt.savefig('lidar in world frame.png')
    return lidar_3d_in_world_frame
    





if __name__ == '__main__':
    run_lidar()