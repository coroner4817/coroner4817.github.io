---
layout: post
title: SLAM Knowledge Tree
date: 2020-11-02 03:23 -0500
categories: [3dvision]
description: >
  Knowledge Tree for SLAM
image:
  path: "/assets/img/blog/slam.jpg"
related_posts: []
---

This post records knowledge structure for the 3D Computer Vision. This post will emphasis on the SLAM technology, which is machine perception and mapping in unknown environment. There is another post that emphasis on Structure from Motion, which talks more about some fundamental concepts [SfM Knowledge Tree](/3dvision/2020-11-02-sfm-knowledge-tree/). 

* toc
{:toc .large-only}

### SLAM Overview
1. Visual Odometry
2. Sensor Fusion
3. Backend Optimization
4. Loop Closure
5. Mapping

A overall SLAM system design can be see in this ORB-SLAM2 diagram:
![ORB-SLAM](/assets/img/blog/orbslam.png)

### Comparison with SfM
1. SLAM require to be real-time while SfM can be offline.
2. SfM also doesn’t require the image data to be in time sequence, while in SLAM data always arrive in sequence.
3. SfM can apply RANSAC across the whole image data set to perform triangulation, while SLAM can only use the data of current frame and previous built map.
4. SfM is batch operation while SLAM is incremental. 
5. SfM can perform non-linear optimization across the whole Bundle Adjustment graph, while SLAM usually only update the partial graph.
6. SfM require accurate mesh reconstruction and texture mapping, while SLAM map can be point cloud or other simplified format.

### Feature based Visual Odometry
1. VO is vision based method which the observed data is the camera data and use it to estimate the hidden camera pose and motion state. Also usually assume the intrinsic matrix is known. 
2. Feature Detection:
	1. ORB
		1. FAST + BRIEF
	2. Subpixel
	3. Feature refinement, when paired features are not on the same Epiplane: 
		1. Minimize Sampson Distance
		2. Mid-point Triangulation
		
3. The solution of recovering the [R, t] is various based on the type of the camera sensor:
	1. Monocular initialization
		1. 2D-2D: Epipolar constraint, Linear algebra solution, 8-points
			1. Essential Matrix
			2. Homogeneous Matrix
	2. Stereo and RGB-D
		1. 2D-3D: PnP
			1. Direct Linear Method
			2. P3P
			3. EPnP
			4. Non-Linear: Bundle Adjustment, minimize reproduction error. 
				1. Construct the least square loss function. 
				2. Calculate the Analytic Partial Derivative (Jacobian)
				3. Use the above linear method output as the initial BA state. 
				4. Optimize both Pose and Landmark
		2. 3D-3D: ICP
			1. Linear method: SVD
			2. Non-Linear: BA
	3. Lidar
		1. Nearest Neighbor

### Optical Flow Visual Odometry
1. Optical Flow can avoid feature detection and feature matching. 
	1. Linear Solution: LK Optical Flow
	2. Non-linear Solution: Direct Method, which also is BA

### Sensor Fusion
1. Use other sensor’s data to constraint the camera motion. E.g. use IMU data for the state update, so that we can better predict the camera pose in the map for PnP
	
### Backend Optimization
1. Backend can refine the front-end rough estimation. It works by define a parametric least square Loss function. And update the state variables to find argmin(state) of the global minimum of the Loss function. The main part of this optimization flow is to find the direction of the gradient to each state variable (Jacobian Matrix) at the current state. Then update the state along that Gradient Descent direction. 
	1. Filter-based method: Extened Kalman Filter
		1. Will have a dedicated Post for KF
	2. Bundle Adjustment: Non-linear Optimization
		1. First order Gradient Descent. 
		2. Second order Gradient Descent (Newton method)
		3. Approximation to the Second Order Gradient Descent
			1. Gaussian-Newton method
			2. Levenberg-Marquardt: Trust region based gradient descent. 

### Loop Closure
1. Bag-of-words
2. Deep Learning based. 

### Mapping 
1. Only insert the feature points on the keyframe to the map
2. Every time detect a loop closure, optimize the whole BA graph

### Misc
1. Left/right Perturbation 
2. Taylor Series
3. Regulation term in loss function: convert the constraint based problem to no constraint problem. 
4. Schur Complement. 
5. Marginalization