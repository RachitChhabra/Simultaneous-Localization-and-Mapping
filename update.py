from more_itertools import time_limited
import numpy as np
import random
import matplotlib.pyplot as plt;
from pr2_utils import read_data_from_csv,bresenham2D, mapCorrelation
from predict import *

res = 0.4
map = np.zeros((int(2000/res),int(2000/res))) + 0.5

resolution = int(500/res)
xy = np.zeros((no_of_particles,3))


filename = 'sensor_data/lidar.csv'
timestamp_lidar,data_lidar = read_data_from_csv(filename)      #  data_lidar ->  (115865, 286), timestamp_lidar (115865,)

timestamp_lidar = (timestamp_lidar-timestamp_lidar[0])/1000000000
angles = np.linspace(-5, 185, 286) / 180 * np.pi

##--------  Lidar r -> xy --------##
lidar_xy = np.zeros((data_lidar.shape[0],data_lidar.shape[1],4))               #  lidar_xy ->  (115865, 286, 4)

lidar_xy[:,:,0] = np.multiply(np.transpose(np.cos(angles)),data_lidar[:,:])
lidar_xy[:,:,1] = np.multiply(np.transpose(np.sin(angles)),data_lidar[:,:])
lidar_xy[:,:,2] = np.zeros(data_lidar.shape)
lidar_xy[:,:,3] = np.ones(data_lidar.shape)

##--------  Lidar xy -> Lidar Vehicle --------##
T_lidar2vehicle   = np.linalg.inv(T_vehicle2lidar)

lidar_2d = lidar_xy.reshape((data_lidar.size,4))                     # reshape for transformaition to vehicle frame (33137390, 4)
lidar_2d = np.matmul(T_lidar2vehicle,lidar_2d.transpose()) # lidar in 2D in vehicle frame
lidar_xy = (np.array(lidar_2d).transpose()).reshape(lidar_xy.shape) # lidar in 3D in vehicle frame 


def update(xy, iteration, alpha):

    lidar_xy_world = get_lidar_coordinates(xy, iteration)       #Provides Lidar coordinates in world frame -  particles x 2 x filtered
 

    alpha_mul = np.zeros((no_of_particles,1))
    for j in range(no_of_particles):
        vp = lidar_xy_world[j]
        x_im = np.arange(0,2000+res,res)
        y_im = np.arange(0,2000+res,res)
        xs = np.arange(-1,1.5,0.5) + xy[j,0]
        ys = np.arange(-1,1.5,0.5) + xy[j,1]

        alpha_mul[j] = np.max(mapCorrelation(map, x_im, y_im, vp, xs, ys))
    
    alpha_int = np.multiply(alpha,alpha_mul)
    alpha = softmax(alpha_int)
    best_particle_index = np.argmax(alpha)
    best_particle = xy[best_particle_index,0:2]
    
    for j in range(lidar_xy_world.shape[2]):
        x_coordinate,y_coordinate = bresenham2D(best_particle[0]/res,best_particle[1]/res,lidar_xy_world[best_particle_index,0,j]/res,lidar_xy_world[best_particle_index,1,j]/res).astype(int)
        x_coordinate = np.array(x_coordinate)
        y_coordinate = np.array(y_coordinate)

        map[-y_coordinate+resolution,x_coordinate+resolution] += np.log(9)

        if(map[-y_coordinate[-1]+resolution,x_coordinate[-1]+resolution]>0):
            map[-y_coordinate[-1]+resolution,x_coordinate[-1]+resolution] -= np.log(64)

    return alpha

def resample(xy, alpha):
    new_alpha = np.zeros(alpha.shape)
    new_xy = np.zeros(xy.shape)
    print(alpha)
    index = random.choice(range(0, no_of_particles-1))
    beta = 0 
    for i in range(no_of_particles):
        beta += random.uniform(0,2*np.max(alpha))
        while alpha[index] < beta:
            beta -= alpha[index]
            index += 1
            if index == 10:
                index = 0
        new_alpha[i] = 1/no_of_particles
        new_xy[i] = xy[index]
    print(new_alpha)
    print('Resampled')
    return new_xy, new_alpha



def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_lidar_coordinates(xy, iteration):           ## xy = [x , y , theta] -> (particles x 3)
    # 1. # Filter Lidar coordinates - Only values 1 < r < 60
    x     = xy[:,0].reshape(no_of_particles,1)
    y     = xy[:,1].reshape(no_of_particles,1)
    theta = xy[:,2].reshape(no_of_particles,1)
    #xy_t = np.transpose(xy)             ## (3 x particles)

    transformation1 = np.stack([np.cos(theta), -np.sin(theta), np.zeros((no_of_particles,1)), x],axis = 2)  ## (particles x 4)
    transformation2 = np.stack([np.sin(theta),  np.cos(theta), np.zeros((no_of_particles,1)), y],axis = 2)  ## (particles x 4)
    transformation = np.hstack((transformation1,transformation2))                                           ## (particles x 2 x 4)

    time_index = argmin(timestamp_lidar,timestamp_encoder[iteration]) 

    lidar_x = lidar_xy[time_index,:,0][((lidar_xy[time_index,:,0]>0.5)&(lidar_xy[time_index,:,0]<60))]
    lidar_y = lidar_xy[time_index,:,1][((lidar_xy[time_index,:,0]>0.5)&(lidar_xy[time_index,:,0]<60))]
    lidar_x = lidar_x.reshape(lidar_x.shape[0],1)
    lidar_y = lidar_y.reshape(lidar_x.shape[0],1)
    lidar_filtered = np.zeros((lidar_x.shape[0],4))

    lidar_filtered = np.hstack((lidar_x,lidar_y,np.zeros((lidar_x.shape[0],1)),np.ones((lidar_x.shape[0],1))))       ##  (filtered x 4)

    lidar_in_world = np.matmul(transformation,np.transpose(lidar_filtered))  ## particles x 2 x filtered
    
    #print(lidar_in_world[0,1,1:10])

    return lidar_in_world


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