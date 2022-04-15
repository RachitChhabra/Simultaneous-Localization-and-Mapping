from tqdm import tqdm
from predict import *
from update import *

N_threshold = 0.2*no_of_particles

alpha = np.zeros((no_of_particles,1))    #(particles,1)
alpha = 1/no_of_particles
xy = np.zeros((no_of_particles,3))
xy_traj = np.zeros((116046,3))


fig = plt.figure()

for i in tqdm(range(data_lidar.shape[0])):
    #PREDICT STEP
    xy_out = predict(xy,i)
    
    if( i % 5 == 0):
        alpha = update(xy,i,alpha)
    xy = xy_out

    trajectory_particle = np.argmax(alpha)
    xy_traj[i,0] =  xy[trajectory_particle,0]/res
    xy_traj[i,1] =  xy[trajectory_particle,1]/res
    xy_traj[i,2] =  xy[trajectory_particle,2]

    ##  ------------- Resampling ------------ ##
    N_eff = 1/np.sum(np.square(alpha))
    if ( N_eff <= N_threshold ):
        xy_out,alpha = resample(xy_out,alpha)


map[(map>-10) & (map!=0.5)] = 0
map[(map==0.5)] = 0.7
map[(map<=-10)] = 1


plt.imshow(map,cmap = 'bone')
plt.scatter(xy_traj[:,0]+int(500/res),-xy_traj[:,1]+int(500/res),color = "r", s = 0.005)
plt.show(block = True)
# cv2.waitKey(30000)


