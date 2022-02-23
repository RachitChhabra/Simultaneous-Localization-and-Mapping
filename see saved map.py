import cv2
import numpy as np
import matplotlib.pyplot as plt
map = np.loadtxt('map1')

cv2.imshow("OCCUPANCY GRID MAP", map)
cv2.waitKey(10000)
