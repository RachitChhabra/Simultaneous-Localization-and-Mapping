import pandas as pd
import cv2, os
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d import Axes3D
import time
from skimage import io


R_vehicle2lidar   = np.matrix('0.00130201 0.796097 0.605167; 0.999999 -0.000419027 -0.00160026; -0.00102038 0.605169 -0.796097')
p_vehicle2lidar   = np.matrix('0.8349; -0.0126869; 1.76416')
T_vehicle2lidar   = np.matrix('0.00130201 0.796097 0.605167 0.8349;0.999999 -0.000419027 -0.00160026 -0.0126869; -0.00102038 0.605169 -0.796097 1.76416; 0 0 0 1')
RPY_vehicle2lidar = np.matrix('142.759; 0.0584636; 89.9254')
T_lidar2vehicle   = np.linalg.inv(T_vehicle2lidar)

R_vehicle2fog   = np.matrix('1 0 0; 0 1 0; 0 0 1')
p_vehicle2fog   = np.matrix('1 0 0 -0.335; 0 1 0 -0.035; 0 0 1 0.78; 0 0 0 1')
RPY_vehicle2fog = np.matrix('-0.335; -0.035; 0.78')
T_vehicle2fog   = np.matrix('0 0 0')

def tic():
  return time.time()
def toc(tstart, name="Operation"):
  print('%s took: %s sec.\n' % (name,(time.time() - tstart)))

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
  stereo = cv2.StereoBM_create(numDisparities = 32, blockSize = 5) 
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
  

def read_data_from_csv(filename):
  '''
  INPUT 
  filename        file address

  OUTPUT 
  timestamp       timestamp of each observation
  data            a numpy array containing a sensor measurement in each row
  '''
  data_csv = pd.read_csv(filename, header=None)#,nrows = 1000)
  data = data_csv.values[:, 1:]
  timestamp = data_csv.values[:, 0]
  return timestamp, data


def mapCorrelation(im, x_im, y_im, vp, xs, ys):
  '''
  INPUT 
  im              the map 
  x_im,y_im       physical x,y positions of the grid map cells
  vp[0:2,:]       occupied x,y positions from range sensor (in physical unit)  
  xs,ys           physical x,y,positions you want to evaluate "correlation" 

  OUTPUT 
  c               sum of the cell values of all the positions hit by range sensor
  '''
  nx = im.shape[0]
  ny = im.shape[1]
  xmin = x_im[0]
  xmax = x_im[-1]
  xresolution = (xmax-xmin)/(nx-1)
  ymin = y_im[0]
  ymax = y_im[-1]
  yresolution = (ymax-ymin)/(ny-1)
  nxs = xs.size
  nys = ys.size
  cpr = np.zeros((nxs, nys))
  for jy in range(0,nys):
    y1 = vp[1,:] + ys[jy] # 1 x 1076
    iy = np.int16(np.round((y1-ymin)/yresolution))
    for jx in range(0,nxs):
      x1 = vp[0,:] + xs[jx] # 1 x 1076
      ix = np.int16(np.round((x1-xmin)/xresolution))
      valid = np.logical_and( np.logical_and((iy >=0), (iy < ny)), \
			                        np.logical_and((ix >=0), (ix < nx)))
      cpr[jx,jy] = np.sum(im[ix[valid],iy[valid]])
  return cpr


def bresenham2D(sx, sy, ex, ey):
  '''
  Bresenham's ray tracing algorithm in 2D.
  Inputs:
	  (sx, sy)	start point of ray
	  (ex, ey)	end point of ray
  '''
  sx = int(round(sx))
  sy = int(round(sy))
  ex = int(round(ex))
  ey = int(round(ey))
  dx = abs(ex-sx)
  dy = abs(ey-sy)
  steep = abs(dy)>abs(dx)
  if steep:
    dx,dy = dy,dx # swap 

  if dy == 0:
    q = np.zeros((dx+1,1))
  else:
    q = np.append(0,np.greater_equal(np.diff(np.mod(np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1,-dy),dx)),0))
  if steep:
    if sy <= ey:
      y = np.arange(sy,ey+1)
    else:
      y = np.arange(sy,ey-1,-1)
    if sx <= ex:
      x = sx + np.cumsum(q)
    else:
      x = sx - np.cumsum(q)
  else:
    if sx <= ex:
      x = np.arange(sx,ex+1)
    else:
      x = np.arange(sx,ex-1,-1)
    if sy <= ey:
      y = sy + np.cumsum(q)
    else:
      y = sy - np.cumsum(q)
  return np.vstack((x,y))
    

def test_bresenham2D():
  import time
  sx = 0
  sy = 1
  print("Testing bresenham2D...")
  r1 = bresenham2D(sx, sy, 10, 5)
  r1_ex = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10],[1,1,2,2,3,3,3,4,4,5,5]])
  r2 = bresenham2D(sx, sy, 9, 6)
  r2_ex = np.array([[0,1,2,3,4,5,6,7,8,9],[1,2,2,3,3,4,4,5,5,6]])	
  if np.logical_and(np.sum(r1 == r1_ex) == np.size(r1_ex),np.sum(r2 == r2_ex) == np.size(r2_ex)):
    print("...Test passed.")
  else:
    print("...Test failed.")

  # Timing for 1000 random rays
  num_rep = 1000
  start_time = time.time()
  for i in range(0,num_rep):
    x,y = bresenham2D(sx, sy, 500, 200)
  print(x,y)
  print("1000 raytraces: --- %s seconds ---" % (time.time() - start_time))


def test_mapCorrelation():
  _, lidar_data = read_data_from_csv('sensor_data/lidar.csv')
  angles = np.linspace(-5, 185, 286) / 180 * np.pi
  ranges = lidar_data[0, :]

  # take valid indices
  indValid = np.logical_and((ranges < 80),(ranges> 0.1))
  ranges = ranges[indValid]
  angles = angles[indValid]

  # init MAP
  MAP = {}
  MAP['res']   = 0.1 #meters
  MAP['xmin']  = -50  #meters
  MAP['ymin']  = -50
  MAP['xmax']  =  50
  MAP['ymax']  =  50 
  MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
  MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
  MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8
  
  #import pdb
  #pdb.set_trace()
  
  # xy position in the sensor frame
  xs0 = ranges*np.cos(angles)
  ys0 = ranges*np.sin(angles)
  
  # convert position in the map frame here 
  Y = np.stack((xs0,ys0))
  
  # convert from meters to cells
  xis = np.ceil((xs0 - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
  yis = np.ceil((ys0 - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
  
  # build an arbitrary map 
  indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < MAP['sizex'])), (yis < MAP['sizey']))
  MAP['map'][xis[indGood],yis[indGood]]=1
      
  x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) #x-positions of each pixel of the map
  y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) #y-positions of each pixel of the map

  x_range = np.arange(-0.4,0.4+0.1,0.1)
  y_range = np.arange(-0.4,0.4+0.1,0.1)


  
  print("Testing map_correlation with {}x{} cells".format(MAP['sizex'],MAP['sizey']))
  ts = tic()
  c = mapCorrelation(MAP['map'],x_im,y_im,Y,x_range,y_range)
  print(c)
  toc(ts,"Map Correlation")

  c_ex = np.array([[ 4.,  6.,  6.,  5.,  8.,  6.,  3.,  2.,  0.],
                   [ 7.,  5., 11.,  8.,  5.,  8.,  5.,  4.,  2.],
                   [ 5.,  7., 11.,  8., 12.,  5.,  2.,  1.,  5.],
                   [ 6.,  8., 13., 66., 33.,  4.,  3.,  3.,  0.],
                   [ 5.,  9.,  9., 63., 55., 13.,  5.,  7.,  4.],
                   [ 1.,  1., 11., 15., 12., 13.,  6., 10.,  7.],
                   [ 2.,  5.,  7., 11.,  7.,  8.,  8.,  6.,  4.],
                   [ 3.,  6.,  9.,  8.,  7.,  7.,  4.,  4.,  3.],
                   [ 2.,  3.,  2.,  6.,  8.,  4.,  5.,  5.,  0.]])
    
  if np.sum(c==c_ex) == np.size(c_ex):
    print("...Test passed.")
  else:
    print("...Test failed. Close figures to continue tests.")

  #plot original lidar points
  fig1 = plt.figure()
  plt.plot(xs0,ys0,'.k')
  plt.xlabel("x")
  plt.ylabel("y")
  plt.title("Laser reading")
  plt.axis('equal')

  #plot map
  fig2 = plt.figure()
  plt.imshow(MAP['map'],cmap="hot");
  plt.title("Occupancy grid map")
  
  #plot correlation
  fig3 = plt.figure()
  ax3 = fig3.gca(projection='3d')
  X, Y = np.meshgrid(np.arange(0,9), np.arange(0,9))
  ax3.plot_surface(X,Y,c,linewidth=0,cmap=plt.cm.jet, antialiased=False,rstride=1, cstride=1)
  plt.title("Correlation coefficient map")

  plt.show()
  plt.savefig('Correlation coefficient map.png')

  
  
def show_lidar():
  _, lidar_data = read_data_from_csv('sensor_data/lidar.csv')
  angles = np.linspace(-5, 185, 286) / 180 * np.pi
  ranges = lidar_data[0, :]
  plt.figure()
  ax = plt.subplot(111, projection='polar')
  ax.plot(angles, ranges)
  ax.set_rmax(80)
  ax.set_rticks([0.5, 1, 1.5, 2])  # fewer radial ticks
  ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
  ax.grid(True)
  ax.set_title("Lidar scan data", va='bottom')
  plt.show()

def lidar_to_vehicle_transformation(data):
  lidar_angle = np.zeros((286,1))
  lidar_angle[0,0] = -5
  for i in range (1,286):
      lidar_angle[i,0] = lidar_angle[i-1] + (0.666)

  lidar_angle = np.deg2rad(lidar_angle)                   # Degree to Radian
  #data[data > 20] = 0
  x = np.multiply(np.transpose(np.cos(lidar_angle)),data[:,:])
  y = np.multiply(np.transpose(np.sin(lidar_angle)),data[:,:])


  lidar_3d = np.stack([x,y,np.zeros(data.shape),np.ones(data.shape)],axis = 2)  #Lidar in 3D in lidar frame (115865, 286, 4)
  lidar_2d = lidar_3d.reshape((data.size,4))                     # reshape for transformaition to vehicle frame (33137390, 4)
  
  lidar_2d_in_vehicle_frame = np.matmul(T_lidar2vehicle,lidar_2d.transpose()) # lidar in 2D in vehicle frame
  lidar_3d_in_vehicle_frame = (np.array(lidar_2d_in_vehicle_frame).transpose()).reshape(lidar_3d.shape) # lidar in 3D in vehicle frame


  return lidar_3d_in_vehicle_frame



if __name__ == '__main__':
  #compute_stereo()
  #show_lidar()
  #test_mapCorrelation()
  test_bresenham2D()
  #buildmap()


  
