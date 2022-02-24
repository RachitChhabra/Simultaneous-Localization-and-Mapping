import math
from re import A
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d import Axes3D
import time
from pr2_utils import read_data_from_csv

##------  Initialise Rotation Matrices  ------##

R_vehicle2lidar   = np.matrix('0.00130201 0.796097 0.605167; 0.999999 -0.000419027 -0.00160026; -0.00102038 0.605169 -0.796097')
p_vehicle2lidar   = np.matrix('0.8349; -0.0126869; 1.76416')
T_vehicle2lidar   = np.matrix('0.00130201 0.796097 0.605167 0.8349;0.999999 -0.000419027 -0.00160026 -0.0126869; -0.00102038 0.605169 -0.796097 1.76416; 0 0 0 1')
RPY_vehicle2lidar = np.matrix('142.759; 0.0584636; 89.9254')

R_vehicle2stereo   = np.matrix('-0.00680499 -0.0153215 0.99985; -0.999977 0.000334627 -0.00680066; -0.000230383 -0.999883 -0.0153234')
p_vehicle2stereo   = np.matrix('1.64239; 0.247401; 1.58411')
T_vehicle2stereo   = np.matrix('-0.00680499 -0.0153215 0.99985 1.64239; -0.999977 0.000334627 -0.00680066 0.247401; -0.000230383 -0.999883 -0.0153234 1.58411; 0 0 0 1')
RPY_vehicle2stereo = np.matrix('-90.878; 0.0132; -90.3899')

R_vehicle2fog   = np.matrix('1 0 0; 0 1 0; 0 0 1')
p_vehicle2fog   = np.matrix('-0.335; -0.035; 0.78')
T_vehicle2fog   = np.matrix('1 0 0 -0.335; 0 1 0 -0.035; 0 0 1 0.78; 0 0 0 1')
RPY_vehicle2fog = np.matrix('0 0 0')


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

##--------  ENCODER  --------##
# The encoder data is stored as [timestamp, left count, right count].
# Encoder calibrated parameter
# Encoder resolution: 4096
# Encoder left wheel diameter: 0.623479
# Encoder right wheel diameter: 0.622806
# Encoder wheel base: 1.52439

##--------  Get data from CSV files  --------##

# Encoder Data
filename = 'sensor_data/encoder.csv'
timestamp,data = read_data_from_csv(filename)

timestamp = (timestamp-timestamp[0])/1000000000

n = timestamp.shape[0]
time_encoder = np.reshape(timestamp,(n,1))
time_encoder_final = time_encoder[1:n]
v=np.zeros((n))
v_avg=np.zeros((n))

# Velocity of the robot.
v = (math.pi*0.6234792*(data[1:n,:]-data[0:n-1,:]))/(4096)#*(time_encoder[1:n]-time_encoder[0:n-1]))
v_left = np.reshape(v[:,0],(n-1,1))

# FOG Data
filename = 'sensor_data/fog.csv'
timestamp,data = read_data_from_csv(filename)
timestamp = (timestamp-timestamp[0])/1000000000
f = timestamp.size

time_fog = np.reshape(timestamp,(f,1))

delta_yaw = np.zeros((f-1))                    # Initialise delta_yaw
delta_yaw = np.reshape(data[1:f,2],(f-1,1))    # Values in radians

delta_yaw_cum = np.reshape(np.cumsum(delta_yaw),(delta_yaw.shape[0],1)) #Cumulative delta_yaw

delta_yaw_10 = delta_yaw_cum[10::10]               # Every 10th element
delta_yaw_10 = delta_yaw_10[0:n-1,:]           # Making equal to no. of Encoder values


time_fog_10 = time_fog[10::10]               # Every 10th element
time_fog_10 = time_fog_10[0:n-1,:]           # Making equal to no. of Encoder values

##-------- Trajectory ---------##
x = np.zeros(n)
y = np.zeros(n)

x = np.multiply(v_left,np.cos(delta_yaw_10))
y = np.multiply(v_left,np.sin(delta_yaw_10))
x_cum = np.cumsum(x)                           # (116047,)
y_cum = np.cumsum(y)                            #(116047,)

def get_vehicle_pos():
    return x_cum,y_cum


def vehicle_to_world_transformation(lidar_vehicle,t):
  T_vehicle2world = np.zeros((lidar_vehicle.shape[0],4,4))    # (115865, 4, 4)
  lidar_world = np.zeros(lidar_vehicle.shape)   #(115865 x 286 x 4)
  for i in range(0,lidar_vehicle.shape[0]):
      time_index = argmin(time_fog,t[i])
      T_vehicle2world[i] = [[np.cos(delta_yaw_cum[time_index]), -np.sin(delta_yaw_cum[time_index]), 0, x_cum[i]], [np.sin(delta_yaw_cum[time_index]), np.cos(delta_yaw_cum[time_index]), 0, y_cum[i]], [0, 0, 1, 1], [0, 0, 0, 1]]
      lidar_world[i] = (np.matmul(T_vehicle2world[i],lidar_vehicle[i].transpose())).transpose()
  return lidar_world


def vehicle_to_world(lidar_vehicle,t,index):
    #T_vehicle2world = np.zeros((4,4))    # (4, 4)
    #lidar_world = np.zeros(lidar_vehicle.shape)  #(100x4)
    time_index = argmin(time_fog_10,t[index])
    T_vehicle2world = [[np.cos(delta_yaw_cum[time_index]), -np.sin(delta_yaw_cum[time_index]), 0, x_cum[index]], [np.sin(delta_yaw_cum[time_index]), np.cos(delta_yaw_cum[time_index]), 0, y_cum[index]], [0, 0, 1, 1], [0, 0, 0, 1]]
   
    lidar_world = (np.matmul((T_vehicle2world),(lidar_vehicle).transpose())).transpose()

    return lidar_world



def argmin(array,value):

    n = len(array)
    if (value < array[0]):
        return -1
    elif (value > array[n-1]):
        return n
    jl = 0# Initialize lower
    ju = n-1# and upper limits.
    while (ju-jl > 1):# If we are not yet done,
        jm=(ju+jl) >> 1# compute a midpoint with a bitshift
        if (value >= array[jm]):
            jl=jm# and replace either the lower limit
        else:
            ju=jm# or the upper limit, as appropriate.
        # Repeat until the test condition is satisfied.
    if (value == array[0]):# edge cases at bottom
        return 0
    elif (value == array[n-1]):# and top
        return n-1
    else:
        return jl