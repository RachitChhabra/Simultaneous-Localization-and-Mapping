import numpy as np, cv2, os
import matplotlib.pyplot as plt

# A = np.matrix('2.7; 1; 2; 5; 3;67;7;43;8;3;6;9;3;2;6')
# B = np.matrix('1;1;0;2;-2;2;4;2;-1;2;1;-5;7;2;3')
# f1 = np.load('f1_R_not_reversed.npy')
# f2 = np.load('f2_R_not_reversed.npy')
# img = cv2.imread('stereo_images/stereo_left/1544583448744279166.png',0)
# img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2RGB)

# textured_img = np.zeros((1500,2000,3))


# for i in range(0,560):
#     for j in range(0,1280):
#         if((f1[800,i,j]<10000)&(f1[800,i,j]>-10000)):
#             if((f2[800,i,j]<10000)&(f2[800,i,j]>-10000)):
#                 textured_img[-int(f2[800,i,j])+100,int(f1[800,i,j])+100] = img[i,j]

a = np.load('particle_trajectory.npy')

plt.plot(a[:,0],a[:,1])
plt.show(block = True)
plt.axis("off")

# A = np.zeros((6,2))
# A[0] = [1,2]
# A[1] = [3,8]
# A[2] = [11,12]
# A[3] = [5,6]
# A[4] = [7,8]
# A[5] = [9,10]

# print(max(A))
