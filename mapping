import math
from operator import index
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d import Axes3D
import time
from pr2_utils import read_data_from_csv,lidar_to_vehicle_transformation, tic,toc,bresenham2D
from trajectory import vehicle_to_world_transformation,get_vehicle_pos
from lidar import run_lidar
start_time=tic()

# MAP = {}
# MAP['res']   = 0.5 #meters
# MAP['xmin']  = 0  #meters
# MAP['ymin']  = -50
# MAP['xmax']  =  50
# MAP['ymax']  =  50 
# MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
# MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
# MAP['map']   = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8

res = int(1)
map = np.zeros((int(2000/res),int(2000/res))) + 0.5


x_cum, y_cum = get_vehicle_pos()



lidar_3d_in_world_frame=run_lidar()

for i in range(0,5000): #lidar_3d_in_world_frame.shape[0]):
    for j in range(0,20):
        x_coordinate,y_coordinate = bresenham2D(np.asscalar(x_cum[i])/res,np.asscalar(y_cum[i])/res,np.asscalar(lidar_3d_in_world_frame[i,j,0])/res,np.asscalar(lidar_3d_in_world_frame[i,j,1])/res)
        
        x  = x_coordinate.astype(int)+int(500/res)
        y  = -y_coordinate.astype(int)+int(500/res)

        #xy = np.stack((y,x),axis = 1)
        map[y,x] = 1
        #print(xy[0])

        # for k  in range(0,x_coordinate.shape[0]):
        #     x = np.int(x_coordinate[k])+500
        #     y = -np.int(y_coordinate[k])+500
        #     map[y,x] = 1
np.savetxt('map1',map)
cv2.imshow("OCCUPANCY GRID MAP", map)
cv2.waitKey(20000)




















toc(start_time)