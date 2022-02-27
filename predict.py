from base64 import encode
import math
from operator import index
from tqdm import tqdm
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d import Axes3D
import time
from pr2_utils import read_data_from_csv,lidar_to_vehicle_transformation, tic,toc,bresenham2D
from trajectory import vehicle_to_world_transformation,get_vehicle_pos,vehicle_to_world
from lidar import run_lidar

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

##--------  Get data from CSV files  --------##

# Encoder Data
filename = 'sensor_data/encoder.csv'
timestamp_encoder,data_encoder = read_data_from_csv(filename)     #data_encoder->(116048, 2), timestamp_encoder -> (116048,)

timestamp_encoder = (timestamp_encoder-timestamp_encoder[0])/1000000000

##--------  Number of particles  --------##
no_of_particles = 100                     ########################################### NO OF PARTICLES ##############################


##--------  Making N Particles  --------##
encoder_clicks = np.zeros((data_encoder.shape[0]-2,no_of_particles))   #(116046, 1)
difference = (data_encoder[1:data_encoder.shape[0],0]-data_encoder[0:data_encoder.shape[0]-1,0])
encoder_clicks[:,:] = np.reshape(difference[1:difference.shape[0]],(encoder_clicks.shape[0],1))

##--------  Addition of Noise in Encoder --------##
noise = np.random.normal(0,20,(encoder_clicks.shape[0],no_of_particles))   #(116046, particles)

encoder_clicks+=noise

##--------  Calculation of movement every timestep without yaw angle consdieration  --------##
v = np.zeros((encoder_clicks.shape[0],1))
v = (math.pi*0.6234792*(encoder_clicks))/(4096)                     #(116046, particles)

# FOG Data - To calculate the delta yaw 
filename = 'sensor_data/fog.csv'
timestamp_fog,data_fog = read_data_from_csv(filename)      #data_encoder->(1160508, 2), timestamp_fog -> (1160508,)
timestamp_fog = (timestamp_fog-timestamp_fog[0])/1000000000
f = timestamp_fog.size
time_fog = np.reshape(timestamp_fog,(f,1))
delta_yaw = np.zeros((f-1))                    # Initialise delta_yaw
delta_yaw = np.reshape(data_fog[1:f,2],(f-1,1))    # Values in radians
delta_yaw_cum = np.reshape(np.cumsum(delta_yaw),(delta_yaw.shape[0],1)) #Cumulative delta_yaw
delta_yaw_10 = delta_yaw_cum[10::10][0:v.shape[0],:]      # Every 10th element - delta_yaw_10 -> (116046, 1) Making equal to no. of Encoder values
time_fog_10 = time_fog[10::10][0:v.shape[0],:]             # Every 10th element  time_fog_10  -> (116046, 1) Making equal to no. of Encoder values

yaw_n = np.zeros((delta_yaw_10.shape[0],no_of_particles))   #(116046, 1)
yaw_n[:,:] = np.reshape(delta_yaw_10[0:delta_yaw_10.shape[0]],(yaw_n.shape[0],1))

##--------  Addition of Noise in Yaw Angle --------##
noise_yaw = np.random.normal(0,0.01,(yaw_n.shape[0],no_of_particles))   #(116046, particles)
yaw_n+=noise_yaw

state_vector = np.stack((np.multiply(v,np.cos(yaw_n)),np.multiply(v,np.sin(yaw_n)),yaw_n),axis = 2)  #(116046, particles, 3)


trajectory = (np.cumsum(state_vector, axis=0))


# ##--------  Visualise Graph --------##
# for i in range(0,no_of_particles):
#     plt.scatter(trajectory[:,i,0],trajectory[:,i,1],s=0.01,color = "r")
    
# plt.show(block = True)

xy_new        = np.zeros((no_of_particles,3))

##--------  Predict the next step  --------##
def predict(xy_pre, iteration): 
    xy_new[:,0] = xy_pre[:,0] + state_vector[iteration,:,0]
    xy_new[:,1] = xy_pre[:,1] + state_vector[iteration,:,1]
    xy_new[:,2] = state_vector[iteration,:,2]
    return xy_new


