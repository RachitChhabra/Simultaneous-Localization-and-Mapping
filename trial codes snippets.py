import numpy as np
from pr2_utils import read_data_from_csv,lidar_to_vehicle_transformation, tic,toc,bresenham2D
from trajectory import vehicle_to_world_transformation,get_vehicle_pos
from lidar import run_lidar
import matplotlib.pyplot as plt

#A = np.matrix('2.7; 1; 2; 5; 3;67;7;43;8;3;6;9;3;2;6')
B = np.matrix('1;1 ;2;2;2;2;2;2;2;2;3;5;7;2;3')


A = np.zeros((6,2))
A[0] = [1,2]
A[1] = [3,4]
A[2] = [11,12]
A[3] = [5,6]
A[4] = [7,8]
A[5] = [9,10]

x = np.matrix('0,0,0')
y = np.matrix('1,2,3')
i = np.stack((x,y),axis = 0)
A[i] = 0
A = np.meshgrid(2,4)
print(A)
