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

> Updated on Mar 28, 2021

### SLAM Overview
1. Visual Odometry
2. Sensor Fusion
3. Backend Optimization
4. Loop Closure
5. Mapping

A overall SLAM system design can be see in this ORB-SLAM2 diagram:
![ORB-SLAM](/assets/img/blog/orbslam.png)

### SLAM Comparison with SfM
1. SLAM require to be real-time while SfM can be offline.
2. SfM also doesn’t require the image data to be in time sequence, while in SLAM data always arrive in sequence.
3. SfM can apply RANSAC across the whole image data set to perform triangulation, while SLAM can match the features between current frame and previous reference frame or built map.
4. SfM is batch operation while SLAM is incremental. 
5. SfM can perform non-linear optimization across the whole Bundle Adjustment graph, while SLAM usually only update the partial graph.
6. SfM 3D Reconstruction require accurate mesh reconstruction and texture mapping, while SLAM map can be point cloud or other simplified format.

### Feature based Visual Odometry
1. VO is vision based method which the observed data is the camera data and use it to estimate the hidden camera pose and motion state. Also usually assume the intrinsic matrix is known. 
2. Feature Detection:
	1. ORB
		1. FAST + BRIEF
	2. Subpixel refinement
	3. Feature refinement, when paired features are not on the same Epiplane: 
		1. Minimize Sampson Distance
		2. Mid-point Triangulation
		
3. The solution of recovering the [R, t] is various based on the type of the camera sensor:
	1. Monocular initialization (Need camera pose has translation)
		1. 2D-2D: Epipolar constraint, Linear algebra solution, 8-points algorithm
			1. Essential Matrix
			2. Homogeneous Matrix
	2. Stereo and RGB-D, or image with map
		1. 2D-3D: PnP
			1. Direct Linear Method with SVD, n = 6
			2. P3P, Linear method, n = 3
			3. EPnP, linear Method, n = 4
			4. Non-Linear: Bundle Adjustment, minimize reproduction error.
				1. Construct the least square reprojection error function. 
				2. Calculate the Analytic Partial Derivative for both pose and landmark (Jacobian)
				3. Use the above linear method output as the initial BA state. 
				4. Optimize both Pose and Landmark using the non-linear optimization algorithm like LM
        5. Each edge will update the vertices connect to it for certain amount of epoch until converge.
		2. 3D-3D point cloud: ICP
			1. Linear method: SVD
			2. Non-Linear: BA
	3. Lidar
		1. Nearest Neighbor
4. Keyframe
	1. When initialization, the # of detected feature points need to larger than a threshold to be consider as initialization success
	2. During SLAM, the definition of a keyframe is: the delta of pose [R, t] with the last keyframe is larger than a threshold

### Optical Flow Visual Odometry
1. Optical Flow can avoid feature detection and feature matching. Basic idea is to assume the camera transform across frame is really small. So that we can have the strong assumption: Grayscale invariant. So that each pixel can be considered as a feature point.  
	1. Linear Solution: LK Optical Flow, still need to detect feature at first frame and does the feature tracking. Still rely on PnP etc to recover the pose
	2. Non-linear Solution: Direct Method, which also is BA. Can optimize the pose and the landmark map. Map can be Sparse, Semi-dense, or Dense. This is also used for Dense Reconstruction. 

### Sensor Fusion
1. Use other sensor’s data to constraint the camera motion. E.g. use IMU data for the state update, so that we can better predict the camera pose in the map for PnP
	
### Backend Optimization
1. Backend can refine the front-end rough estimation. It works by define a parametric least square Loss function. And update the state variables to find argmin(state) of the global minimum of the Loss function. The main part of this optimization flow is to find the direction of the gradient to each state variable (Jacobian Matrix) at the current state. Then update the state along that Gradient Descent direction. 
	1. Filter-based method: Extened Kalman Filter
		1. Will post another blog for KF
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
4. Marginalization: Schur Complement. 

---

### Overview
1. General categories
	1. System: Monocular, Stereo, RGB-D, VIO, LiDAR.
	2. VO: 2D-2D, 2D-3D, 3D-3D
	3. LiDAR: scan-scan, scan-map
	4. IMU & GPS: Pre-integration, Difference method
	5. Reconstruction: Sparse, Semi-dense, Dense
	6. Other topics: Object detection, Semantic Segmentation, etc. 
2.  Milestone works
	1. PTAM 2007
	2. DTAM 2011
	3. ORB-SLAM2 2017 
	4. VINS-Mono 2017
	5. DSO 2016
	6. PL-VIO 2018
	7. DL driven:
		1. DVSO 2018
		2. CNN-SLAM 2018
3. Multi-Geometry Features
	1. Feature Points
		1. Versatile to lighting
		2. May lose tracking
	2. Feature Line
	3. Feature Plane
	4. Deep Learning Features
    
### Resources
1. Frameworks/Libraries
	1. OS: ROS
	2. Math: Eigen
	3. CV: OpenCV, PCL
	4. Optimization: g2o, Ceres, gtsam, PyTorch Geometric
	5. Graphic: OpenGL, Pangolin
	6. Features: libpointmatcher, DBow2
	7. Mapping: OctoMap, OpenFABMAP
    
2. Open-source projects
	1. [GitHub - yanyan-li/SLAM-BOOK](https://github.com/yanyan-li/SLAM-BOOK)
	2. [GitHub - wuxiaolang/Visual_SLAM_Related_Research](https://github.com/wuxiaolang/Visual_SLAM_Related_Research)
	3. [GitHub - raulmur/ORB_SLAM2: Real-Time SLAM for Monocular, Stereo and RGB-D Cameras, with Loop Detection and Relocalization Capabilities](https://github.com/raulmur/ORB_SLAM2)
	4. [GitHub - UZ-SLAMLab/ORB_SLAM3: ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM](https://github.com/UZ-SLAMLab/ORB_SLAM3)
	5. [GitHub - colmap/colmap: COLMAP - Structure-from-Motion and Multi-View Stereo](https://github.com/colmap/colmap)
	6. [GitHub - JakobEngel/dso: Direct Sparse Odometry](https://github.com/JakobEngel/dso)
	7. [GitHub - RobustFieldAutonomyLab/LeGO-LOAM: LeGO-LOAM: Lightweight and Ground-Optimized Lidar Odometry and Mapping on Variable Terrain](https://github.com/RobustFieldAutonomyLab/LeGO-LOAM)
	8. [GitHub - mapillary/OpenSfM: Open source Structure-from-Motion pipeline](https://github.com/mapillary/OpenSfM)
	9. [GitHub - tum-vision/lsd_slam: LSD-SLAM](https://github.com/tum-vision/lsd_slam)
	10. [GitHub - HKUST-Aerial-Robotics/VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator](https://github.com/HKUST-Aerial-Robotics/VINS-Mono)

### CV Basics
1. Feature Fitting
	1. RANSAC: Random Sample Consensus
	2. J-Linkage
	3. T-Linkage
2. Feature Points: SIFT, ASIFT, SURF, ORB
	1. SIFT: 
		1. DoG(Difference of Gaussians) to get image information at different frequency level, aka Gaussian Pyramid.  
		2. Find initial feature points in 3-dimensional Gaussian Pyramid, which are the local maximum or minimum of the 26 surrounding points. 
		3. Remove outlier feature points at the edge. Also refine the feature points position
		4. Calculate the Gradient Direction of the feature point based on the 4 directional pixels around it at the same Gaussian Pyramid level. 
		5. Build Descriptor: select the kernel of 16x16 around the key point. For every 4x4 sub-kernel, build the gradient histogram. Each bin represent gradient direction within 45 degree (8 bins). There are 16 sub-kernel. So we have a 128-dim vector descriptor
		6. For larger rotation between 2 frames, we can use ASIFT
	2. SURF: 
		1. Main difference with SIFT is the descriptor
	3. ORB: 
		1. FAST corner point
			1. For all points. Select the 16 points at the circle of radius 3 around the point. 
			2. It is a feature point if there are more than N (usually 12) points on that circle is larger or smaller than the point’s color with a threshold.
			3. Pass the feature point to a decision tree to trim bad features
			4. After iterate across the image, select feature points of top-N Harris Response.
			5. ORB also take care of the Scale-invariant by using Gaussian Pyramid and detect FAST at each level
			6. For Rotation-invariant, ORB use Intensity Centroid. The gradient direction is defined as the direction from feature point to the centroid of a small kernel around it. 
		2. BRIEF descriptor
			1. Select a kernel around the feature point. Random select point pairs within the kernel. Use binary to represent the comparison between the 2 points. So we have a binary 128-dim vector descriptor
			2. Pair Selection can base on 2D Gaussian distribution
			3. Steer BRIEF: apply rotation to the pair selection mask base on the Intensity Centroid calculated in FAST
	4. Feature Points Matching across 2 frames
		1. Brute-force
		2. Hamming Distance
		3. FLANN
3. Feature Line
	1. Hough Transform
	2. LSD: cv::line_descriptor. Note this is not LSD-SLAM
4. Feature Plane
	1. Extracting plane from Point Cloud
		1. Calculate normal for each 3D point, based on the surrounding 3D points and prior info
		2. Clustering points with similar normal direction
	2. AHC: Agglomerative Hierarchical Clustering. A better open-source solution
	3. Plane Matching based on normal + distance between origin and the plane
5. Feature Constraints
	1. Vanishing Point Constraint: parallel lines on the image will be merged at a point, due the perspective projection
		1. Vanishing Point is used for online calibration
	2. Vanishing Line Constraint: a 3D space plane’s intersection line with the camera plane. Any set of lines parallel to the 3D space plane, their vanish point will be on the vanishing line.
	3. Manhattan World Constraint: Our world usually is build with 3 perpendicular planes
	4. Same Plane Constraint: for feature point/line on the same plane, they will maintain the constraint across frames

### Deep Learning
1. Popular Architectures: VGG, ResNet, DenseNet, AlexNet, GoogleNet, MobileNet, U-Net
2. Network design Tips:
	1. 1x1 convolution kernel: use to adjust the tensor size. Whether reduce or increase depend on the number of the filters
	2. Residual module can solve the issue of gradient explosion/diminish
	3. After ReLU, need to add a local response normalization layer to rectify the tensor data. This is normalize the output (feature map) of all neurons in 1 layers for 1 sample
	4. Batch Normalization: normalize the output of same neuron across all the samples in 1 mini batch.
3. Gradient Descent Optimizer
	1. SGD + Momentum
	2. Adaptive Learning rate: Adam, RMSProp, AdaDelta
4. Model Evaluation:
	1. Precision and Recall: ROC, AUC
	2. Accuracy

### Deep Learning in Computer Vision
1. Feature Point Prediction: GCNv2
2. Feature Line Prediction
3. Object Detection
	1. Bounding Box
	2. Instance Segmentation
4. Obtain depth map from monocular frame
	1. FCN: Fully Convolutional Networks for Semantic Segmentation
		1. Deconvolution: Padding original tensor to bigger size and do a normal convolution, such that output is larger than original tensor size
	2. CRF: conditional random field
	3. CNN-SLAM: not real-time
	4. MobileNet V2: decompose convolution to depth-wise convolution and 1x1 convolution
	5. GeoNet: consider normal map and depth map together
	6. Unsupervised Learning:
		1. monodepth, monodepthv2
      1. Training on Stereo camera data, work on Monocular camera
      2. Training a encoder-decoder NN, which aim to generate per-pixel disparity maps (a scalar value per pixel that the model will learn to predict) for both Left image and right image. And the input is ONLY the left image. The network should converge the disparity to the stereo camera baseline. But during this unsupervised training process, the NN learn about the real world scale, distance, object segmentation and other info from the dataset.
      3. The goal of the whole pipeline is to generate right image by shifting the left image pixels, and generate left image by shift right image pixels. 
      4. The shift is done via a pair of Differentiable Image Samplers (Spatial Transformer Network), which one takes left image and right disparity map to generted right image, and the other one takes right image and left disparity map to generted left image
      5. The loss is calculated of the projection error base on output image and groundtruth image, so this is why need stereo camera data. Also add penalty of Left and right disparity map consistency and disparity map smoothness loss 
      6. Finally, we can use the disparity map of the left image to generate per-pixel depth map: depth = (baseline * focal length) / disparity. 
      7. Unsupervised because it ignores the fact that we know the baseline between the camera, and use this as a constraint to training the end-to-end network. Notice that this is still a data-driven DL approach not a generalized CV approach, such that the NN is not learning to estimate camera pose but learning to assign depth to each pixel. The NN might potentially doing object recognition and segmentation etc under the hood ultimately. So still might be overfitting to the dataset.
      8. Thoughts: When inferencing take 2 consecutive monocular frames as input, Maybe use 2D/3D/4D disparity map, NN also intermediate output depth  
5. Rendering
	1. NeRF
6. 3D Data learning
	1. PyTorch3D

### Graph Representation Learning
1. PyTorch Geometric
	1. Graph optimizer like g2o, but using deep learning

### Sensor Fusion
1. Camera 
	1. Intrinsic matrix
	2. Distortion coefficient
2. IMU
3. LiDAR

### Pose Estimation Model
1. Pose constraints models: PnP, ICP, etc. See above
2. Constraints to avoid drifting error
	1. Manhattan World Constraint
		1. Reality world is build with orthogonal structures, like floor, wall, ceiling. Using those as prior info
		2. Use Deep Learning predict the surface normal. Then base on angle difference, clustering the normal. This is called Sphere mean shift
		3. Then estimate the rotation. Drift-free Rotation Estimation
		4. Base on Feature point to estimate translation
	2. Atlanta World Constraint
		1. Multiple combination of the Manhattan World models

### Mapping
1. Sparse Point Cloud map
	1. Feature points pairs triangulation
		1. Select feature points pair based on Parallax angle
			1. Select if the angle between the 2 camera->feature rays is larger than a threshold
		2. Feature points should be first convert from UV space to the normalized camera space
			1. x_camNorm = {u - K[0][2]_K[0][0], v - K[1][2]_K[1][1]}
		3. Then convert from normalized camera space to camera space
			1. x_cam = x_camNorm * (depth of the 3D point)
		4. Then we can use the camera pose of the 2 frame to triangulation. Notice that if the extrinsic matrix are already in the world space (not relative pose), we can skip step 2. Also we can even fuse the intrinsic here
			1. cv::triangulatePoints
				1. [x1_cam.cross(extrinsic1);
         x2_cam.cross(extrinsic2) * X_world = 0
				2. Use SVD to solve this linear equation
			2. Midpoint triangulation
				1. Inverse project the x_cam1 and x_cam2 to X_world1 and X_world2. Then use the mid-point of X_world1 and X_world2 as result.
				2. Solve linear equation using Cramer’s rule
		5. Normalize the Triangulation output
			1. PWorld = PTri[0:2] / PTri[3]
		6. Remove Outliner
			1. Statistical Filtering: calculate the distribution for each point’s distance to the NN. Then set the upper and lower bound threshold base on the distribution and remove outliners.
			2. Voxel Filter: reduce the point in each voxel to be 1
		7. Mesh Generation (optional)
			1. Marching Cubes to generate mesh based on point cloud  
		8. Point cloud Map data structure
			1. OctoMap: like the BVH, a fast way to find out occupancy info of a certain space
	2. Feature Line Reconstruction
		1. Similar Parallax angle selection criteria as above 
		2. Similar to feature points triangulation, but here for both lines select a start points matching pair and an end points matching pair. Then we build the linear equation as above and use SVD solve it
		3. Then solve the line equation using the 2 points on the line: X_start_world and X_end_world
2. Dense Map Reconstruction
	1. DTAM: Monocular Optical flow tracking dense reconstruction on GPU
		1. Use Optical Flow Direct Method (Brightness Constancy Assumption) to tracking the pose and optimize the landmark.
		2. If the landmark (pixel) is across the whole image dimension, then the Direct Method output is the Dense Reconstruction
	2. Dense map representation
		1. Mesh: build vertices mesh base on point cloud
		2. TSDF: Truncated Signed Distance Function
			1. For each voxel, store the distance to the closet surface. If in front of the surface, the value is positive. Else value is negative. 
				1. TSDF is relative to the camera center at each frame, so the distance can be roughly define as the pixel_depth - surface_depth.
				2. TSDF has a cutoff threshold 
				3. Also assign a weight for each voxel, based on angle to camera.
				4. When a new frame added to the scene, use an update function to update the TSDF and wight for the matched voxel. 
			2. Marching Cubes to obtain the surface. 
				1. This is running every time when a new frame is added.
				2. After obtain the voxel map with TSDF value, use Marching Cubes of certain size to walk the whole voxel map. If the sum of TSDF value within the cube is 0, then we consider this is a point on surface.
				3. Eventually, we can use use Marching cube again to build the mesh
		3. Surfel
      
### Non-linear Optimization
1.  Bundle Adjustment
	1. Build pose graph base on constraints. Minimize the least square loss of the graph
		1. Cost function: for different type of features, build the parametric cost function using pose and landmark
			1. Point, Line, Plane feature
		2. Calculate the Jacobian matrix: first order partial derivate
		3. We can use multiple type of feature together to build a cost function
	2. Update Method
		1. Gaussian-Newton: need good initial value
		2. Leverberg-Marquart: need more iteration
		3. BFGS: a better solution

### SLAM framework in depth
1. ORB-SLAM2
	1. 4 modules: Front-end, Mapping, Loop detection, Loop closure
	2. 3 threads
	3. Initialization
		1. Stereo and RGB-D share a initialization flow, while monocular has it own flow
		2. Monocular initialization
			1. If find 2 consecutive frames which number of matched feature point pair more than a certain threshold, estimate the pose between these 2 frame
			2. When estimate, calculate both the Fundamental and Homograph matrix. Homograph matrix is more for the case of most feature points falling on a plane, or small transformation.
			3. Use RANSAC and 8-point algorithm to estimate both Fundamental and Homograph. Calculate the loss ratio: R_H = S_H / (S_H + F_H). If R_H > 0.045, choose Homograph matrix for the initial frame 
		3. Pose Estimation and optimization
			1. Evenly select feature by dividing the image frame in to different ROI
			2. Obtain feature world position
				1. For monocular, we need to triangulation to obtain 3D position
				2. For stereo and RGB-D, we can directly obtain the 3D position
			3. 6D pose estimation: For 2 new frame pair, given an initial value for the pose and optimize the reprojection cost function. Then also update the local partition of the map. If is keyframe, insert to map and update the whole map in another thread
				1. frame-to-frame 
				2. frame-to-map (aka keyframes collection)
		4. Bag-of-Word Loop detection
			1. To avoid drifting
			2. Using DBoW2
				1. Unsupervised Train a feature dictionary using prior data. Like Word2Vec
				2. For each frame, find all the features and corresponding in the dictionary, then use the histogram descriptor to describe the frame
				3. In the map, find a keyframe that can partially match this descriptor and consider as loop detection
				4. Add an new edge in pose graph and optimize the whole pose graph
				5. For monocular, we also need to calculate the sim(3) 3D Similarity transformations, to optimize the scale for the whole pose graph
2. LSD-SLAM: Large-Scale Direct Monocular SLAM
	1. Use Optical Flow Direct method to localization. So that doesn’t need to do feature detection. Work better when features are not enough
	2. Also do Semi-dense map building on a larger scale
	3. Algo Flow
		1. Image tracking: find the rigid body transformation between current frame and last frame
		2. Depth Estimation: NCC (Normalized Cross Correlation). Check slambook
		3. Map Optimization: also check for loop closure base on image similarity 
3. SVO: Semi-Direct Monocular Visual Odometry
	1. Use both sparse feature points and Optical Flow direct method
		1. Feature for loop detection
		2. Optical Flow for pose estimation
	2. Pose Estimation thread
	3. Depth Estimation thread
		1. Gaussian-uniformly mixed distribution depth filter
    
### Deep Learning SLAM
1. Structure-SLAM
	1. Monocular
	2. Use deep learning (Encoder-Decoder) to estimate the surface normal
	3. Use Manhattan World model to estimate the Rotation. And use feature (ORB) to estimate translation
	4. Refine the pose using local map. 
	5. Update the map if is a keyframe