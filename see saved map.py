import numpy as np
import matplotlib.pyplot as plt
map = np.loadtxt('occupancy_grid2.txt')

rgb_occupancy_grid = np.zeros((2000,2000,3))


for i in range(2000):
    for j in range(2000):
        if map[i,j] == 0:
            rgb_occupancy_grid[i,j] = [0,0,0]
        if map[i,j] == 0.7:
            rgb_occupancy_grid[i,j] = [128,128,128]
        if map[i,j] == 1:
            rgb_occupancy_grid[i,j] = [255,255,255]

np.save('rgb_occupancy_grid.npy',rgb_occupancy_grid)
plt.imshow((rgb_occupancy_grid).astype(np.uint8))

plt.show(block = True)
