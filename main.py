from tqdm import tqdm
from predict import *
from update import *

alpha = np.zeros((no_of_particles,1))    #(particles,1)
alpha = 1/no_of_particles


for i in tqdm(range(116045)):
    #PREDICT STEP
    xy_out = predict(xy,i)
    
    if( i % 5 == 0):
        alpha = update(xy,i,alpha)

    xy = xy_out

# map[(map<1) & (map!=0.5)] = -1
np.savetxt('map_main.txt',map)
cv2.imshow("OCCUPANCY GRID MAP", map)
cv2.waitKey(30000)
