# Simultaneous-Localization-and-Mapping

The goal of this project is to create a particle filter simultaneous localization and mapping model using data from an autonomous vehicle.

SLAM, as commonly known in the industry, is a mathematical problem of building a map or an unknown environment,
  and at the same time track the position of an agent (a vehicle in this problem) in this space.
  
SLAM is used in a variety of applications such as autonomous vehicles, virtual and augmented reality applications, etc. 

In this project, using the data provided first dead reckoning is done using one particle, and no added noise to a differential drive model.

Next, using Light Detection and Ranging (LiDAR) data provided, and scan grid correlation observation model,
  SLAM is carried out, and an Occupancy Grid Map is constructed for multiple particles and added gaussian noise to the model.
  
Finally, using the stereo camera model, and given data from the two cameras on the vehicle, a textured map is created of the environment of the vehicle.
