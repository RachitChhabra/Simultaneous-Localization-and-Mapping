import cv2
import numpy as np
import matplotlib.pyplot as plt
map = np.loadtxt('map_particles_100_res_0_5_map_9.txt')

map[(map<6) & (map!=0.5)] = 0.5
map[(map==0.5)] = 0.3   
map[(map>=6)] = 1

# np.savetxt('map_particles_100_res_0_5_map_9_more_noise_resampling_0.5.txt',map)


plt.imshow(map,cmap = 'Greys')

plt.show(block = True)
