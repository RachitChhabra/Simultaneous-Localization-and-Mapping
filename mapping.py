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
start_time=tic()

res = 0.5
map = np.zeros((int(2000/res),int(2000/res))) + 0.5
x_cum, y_cum = get_vehicle_pos()
resolution = int(500/res)

lidar_3d_in_vehicle_frame,timestamp=run_lidar()


for i in tqdm(range(0,lidar_3d_in_vehicle_frame.shape[0])):
    indexValid = np.where(np.logical_and((np.square(lidar_3d_in_vehicle_frame[i,:,0])+np.square(lidar_3d_in_vehicle_frame[i,:,1])<2500),(np.square(lidar_3d_in_vehicle_frame[i,:,0])+np.square(lidar_3d_in_vehicle_frame[i,:,1])>9)))
    
    lidar_values_in_range = lidar_3d_in_vehicle_frame[i,indexValid,:]
    lidar_values_in_range = np.reshape(lidar_values_in_range,(lidar_values_in_range.shape[1],4))

    lidar_values_in_range_in_world = vehicle_to_world(lidar_values_in_range,timestamp,i)

    for j in range(0,lidar_values_in_range_in_world.shape[0]):
        x_coordinate,y_coordinate = (bresenham2D(int(x_cum[i])/res,int(y_cum[i])/res,int(lidar_values_in_range_in_world[j,0])/res,int(lidar_values_in_range_in_world[j,1])/res)).astype(int)
        x_coordinate = np.array(x_coordinate)
        y_coordinate = np.array(y_coordinate)

        map[-y_coordinate+resolution,x_coordinate+resolution] += np.log(4)

        if(map[-y_coordinate[-1]+resolution,x_coordinate[-1]+resolution]>-5):
            map[-y_coordinate[-1]+resolution,x_coordinate[-1]+resolution] -= np.log(8)



map[(map<1) & (map!=0.5)] = -1
toc(start_time)
#np.savetxt('map2.txt',map)
cv2.imshow("OCCUPANCY GRID MAP", map)
cv2.waitKey(30000)
