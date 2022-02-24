import cv2
import numpy as np
import matplotlib.pyplot as plt
map = np.loadtxt('map1')
map[map>10] = 1
map[map<10] = 0
cv2.imshow("OCCUPANCY GRID MAP", map)
cv2.waitKey(10000)
