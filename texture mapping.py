import numpy as np
from tqdm import tqdm
import os, cv2
from update import argmin
from pr2_utils import read_data_from_csv
import matplotlib.pyplot as plt;

fsu = 7.7537235550066748e+02
fsv = 7.7537235550066748e+02
cu  = 6.1947309112548828e+02
cv  = 2.5718049049377441e+02
b   = 0.475143600050775

R_vehicle2stereo   = np.matrix('-0.00680499 -0.0153215 0.99985; -0.999977 0.000334627 -0.00680066; -0.000230383 -0.999883 -0.0153234')
p_vehicle2stereo   = np.matrix('1.64239; 0.247401; 1.58411')
T_vehicle2stereo   = np.matrix('-0.00680499 -0.0153215 0.99985 1.64239; -0.999977 0.000334627 -0.00680066 0.247401; -0.000230383 -0.999883 -0.0153234 1.58411; 0 0 0 1')
RPY_vehicle2stereo = np.matrix('-90.878; 0.0132; -90.3899')

xy_traj = np.load('particle_trajectory_low_noise.npy')

timestamp_lidar,_ = read_data_from_csv('sensor_data/lidar.csv')      #  data_lidar ->  (115865, 286), timestamp_lidar (115865,)



disparity_matrix = np.zeros((1161,560,1280))
file_time = np.zeros((1161,1))
imagex = np.zeros((1161,560,1280))
imagey = np.zeros((1161,560,1280))
imagez = np.zeros((1161,560,1280))
m1 = np.zeros((1161,560,1280))
m2 = np.zeros((1161,560,1280))
m3 = np.zeros((1161,560,1280))

uL = np.zeros((560,1280))
uL[:] += np.linspace(0,1279,1280)

vL = np.zeros((1280,560))
vL[:] += np.linspace(0,559,560)
vL = np.transpose(vL)
res = 1
separator = '.'
maxsplit = 1
folder = 'stereo_images/stereo_right/'

def texture_mapping():

    get_stereo_coordinates()

    return 0

left_dir = 'stereo_images/stereo_left/'
right_dir = 'stereo_images/stereo_right/'
common_images = set.intersection(*(set(os.path.relpath(os.path.join(root, file), path) for root, _, files in os.walk(path) for file in files) for path in (left_dir, right_dir)))

def get_stereo_coordinates():
    image_no = 0

    for filename in tqdm(sorted(common_images)):
        disparity_matrix[image_no] = compute_stereo(filename)
        imagez[image_no] =  (1/disparity_matrix[image_no])*fsu*b
        imagez[image_no][imagez[image_no]>50] = 0
        imagex[image_no] =  (1/fsu)*(uL - cu)*imagez[image_no]
        imagey[image_no] =  (1/fsv)*(vL - cv)*imagez[image_no]

        name,_ = filename.split(separator, maxsplit)
        file_time[image_no] = int(name)

        imagexyz = np.stack((imagex[image_no],imagey[image_no],imagez[image_no]),axis = 2)
        a = np.zeros((560,1280,3))
        for j in range(0,560):
            a[j] = np.transpose(R_vehicle2stereo*np.transpose(imagexyz[j]) + p_vehicle2stereo)
        
        m1[image_no] = a[:,:,0]
        m2[image_no] = a[:,:,1]
        m3[image_no] = a[:,:,2]

        m1231 = np.stack((m1[image_no],m2[image_no],m3[image_no],np.ones((560,1280))),axis = 2)    ## (560, 1280, 4)

        time_index = argmin(timestamp_lidar,file_time[image_no]) 
        transformation1 = np.matrix([np.cos(xy_traj[time_index,2]), -np.sin(xy_traj[time_index,2]), 0, xy_traj[time_index,0]])  ## (1 x 4)
        transformation2 = np.matrix([np.sin(xy_traj[time_index,2]),  np.cos(xy_traj[time_index,2]), 0, xy_traj[time_index,1]])  ## (1 x 4)
        transformation3 = np.matrix([0,0,1,0])      ## (1 x 4)
        transformation4 = np.matrix([0,0,0,1])      ## (1 x 4)

        transformation = np.stack((transformation1,transformation2, transformation3, transformation4))   ## (4 x 4)

        print(transformation)
        c = np.zeros((560,1280,4))
        for j in range(0,560):
            c[j] = np.transpose(transformation*np.transpose(m1231[j]))
                
        m1[image_no] = c[:,:,0]
        m2[image_no] = c[:,:,1]
        m3[image_no] = c[:,:,2]

        image_no+=1

    # stacked = np.stack((m1,m2,m3),axis = 3)
    # print('Transformed')
    # np.save('stereo_in_world',stacked)

    # np.savetxt('Yo.txt',disparity_matrix[disparity_matrix!=0])
    np.save('f1_R_reversed',m1)
    np.save('f2_R_reversed',m2)
    np.save('f3_R_reversed',m3)
    # np.save('stereo_images_timestamp',file_time)
    # np.save('disparity_matrix',disparity_matrix)


def load_data():
    #map     = np.loadtxt('map_particles_100_res_0_5_map_9_more_noise.txt')
    #f1      = np.load('f1.npy')
    f1      = np.load('f1_R_reversed.npy')     
    #f2      = np.load('f2.npy')
    f2      = np.load('f2_R_reversed.npy') 
    f3      = np.load('f3_R_reversed.npy')
    #traj_xy = np.load('particle_trajectory.npy')
    
    # stereo_time_stamp    = np.load('stereo_images_timestamp.npy')
    # for file in tqdm(sorted(path_l)):
    count=0
    textured_img = np.zeros((1500,2000,3))
    for files in tqdm(sorted(common_images)):

        img = cv2.imread(os.path.join(folder,files),0)
        img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)


        # name,_ = files.split(separator, maxsplit)
        # photo_time = int(name)
        # index = argmin(stereo_time_stamp, photo_time)
        
        
        for i in range(560):
            for j in range(1280):
            #   if((f1[count,i,j]<10000)&(f1[count,i,j]>-10000)):
            #         if((f2[count,i,j]<10000)&(f2[count,i,j]>-10000)):
                if((f3[count,i,j]<1)&(f3[count,i,j]>-20)):
                    a = -int(f2[count,i,j])+100
                    b =  int(f1[count,i,j])+100
                    #if(textured_img[a,b].all() == 0):  
                    textured_img[a,b] = img[i,j]

        count+=1

    plt.imshow((textured_img).astype(np.uint8))
    plt.show(block = True)
    plt.axis("off")



def fun():
    load_data()


def compute_stereo(filename):
  # path_l = 'code/data/image_left.png'
  # path_r = 'code/data/image_right.png'

  path_l = 'stereo_images/stereo_left/'
  path_r = 'stereo_images/stereo_right/'

  # image_l = cv2.imread(path_l, 0)
  # image_r = cv2.imread(path_r, 0)

  # image_l = cv2.imread(os.path.join(path_l,filename), 0)
  # image_r = cv2.imread(os.path.join(path_r,filename), 0)

  image_l = cv2.imread(os.path.join(path_l,filename),0)
  image_r = cv2.imread(os.path.join(path_r,filename),0)

  image_l = cv2.cvtColor(image_l, cv2.COLOR_BAYER_BG2BGR)
  image_r = cv2.cvtColor(image_r, cv2.COLOR_BAYER_BG2BGR)

  image_l_gray = cv2.cvtColor(image_l, cv2.COLOR_BGR2GRAY)
  image_r_gray = cv2.cvtColor(image_r, cv2.COLOR_BGR2GRAY)

  # You may need to fine-tune the variables `numDisparities` and `blockSize` based on the desired accuracy
  stereo = cv2.StereoBM_create(numDisparities = 32, blockSize = 9) 
  disparity = stereo.compute(image_l_gray, image_r_gray)

  # fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
  # ax1.imshow(image_l)
  # ax1.set_title('Left Image')
  # ax2.imshow(image_r)
  # ax2.set_title('Right Image')
  # ax3.imshow(disparity, cmap='gray')
  # ax3.set_title(filename)
  # plt.show(block = True)
  return disparity




if __name__ == '__main__':
    texture_mapping()
    #fun()

