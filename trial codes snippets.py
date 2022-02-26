import numpy as np
from pr2_utils import read_data_from_csv,lidar_to_vehicle_transformation, tic,toc,bresenham2D
from trajectory import vehicle_to_world_transformation,get_vehicle_pos
from lidar import run_lidar
import matplotlib.pyplot as plt

#A = np.matrix('2.7; 1; 2; 5; 3;67;7;43;8;3;6;9;3;2;6')
B = np.matrix('1;1;0;2;-2;2;4;2;-1;2;1;-5;7;2;3')

A = np.zeros((6,2))
A[0] = [1,2]
A[1] = [3,8]
A[2] = [11,12]
A[3] = [5,6]
A[4] = [7,8]
A[5] = [9,10]

print(max(A))
