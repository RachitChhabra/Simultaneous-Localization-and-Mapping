import numpy as np
from pr2_utils import read_data_from_csv,lidar_to_vehicle_transformation, tic,toc,bresenham2D
from trajectory import vehicle_to_world_transformation,get_vehicle_pos
from lidar import run_lidar
import matplotlib.pyplot as plt

#A = np.matrix('2.7; 1; 2; 5; 3;67;7;43;8;3;6;9;3;2;6')
# B = np.matrix('1;1;0;2;-2;2;4;2;-1;2;1;-5;7;2;3')

# B[(B<2)&(B!=1)] = 0
# print(B)

# A = np.zeros((6,2))
# A[0] = [1,2]
# A[1] = [3,8]
# A[2] = [11,12]
# A[3] = [5,6]
# A[4] = [7,8]
# A[5] = [9,10]

# A[A<5]=0
# print(A)

# x = np.matrix('0,0,0')
# y = np.matrix('1,2,3')
# i = np.stack((x,y),axis = 0)
# A[i] = 0
# A = np.meshgrid(2,4)
# print(A)
C = np.matrix('1,2,3,4')
A = np.random.randn(1,2,3)

print(A.shape)
# particles = 5
# v = np.zeros((particle,timestamp))