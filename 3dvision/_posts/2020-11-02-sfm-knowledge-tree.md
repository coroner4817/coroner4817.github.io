---
layout: post
title: Structure from Motion Knowledge Tree
date: 2020-11-02 01:44 -0500
categories: [3dvision]
description: > 
  Knowlegde Tree for SfM
image:
  path: "/assets/img/blog/sfm.png"
related_posts: []
---

This post is about the fundamental knowledge and concepts in the 3D Computer Vision field. The knowledge points of this post is mainly base on the book [An Invitation to 3-D Vision SpringerLink](https://link.springer.com/book/10.1007/978-0-387-21779-6). I will try to rephrase the knowledge using my words and combine with some other knowledges. 

* toc
{:toc .large-only}

### Linear Space
1. Linear space / Vector space: Given any vectors within the space, they are always closed within the space under any linear combinations.
	1. Subspace: A subset of the Linear space. Also it is linear closed and contain the zero vector
	2. Spanned subspace: Expand a subspace to a bigger vector space which contains all possible linear combinations of the original subspace.
	3. Linear independence: A set of vector is linear independence when the linear combination of the vectors only equal 0 when scalar parameters are 0s. E.g. One-hot encoding. 
	4. Basis: A set of linear independent vector and they can span to a linear space. 
		1. The number of the basis is always equal the number of the dimension of the spanned linear space.
		2. The change-of-basis matrix: There are 2 scenarios: 
			1. the new basis is wrote in the linear combination of the original basis. 
				1. In this situation, the original basis is consider as the standard basis. So the change-of-basis matrix can be just expressed as: each new basis vector as each row. So the using this change-of-basis multiply the original coordinates means projecting the old coordinate to the new basis. A example usage is the TBN matrix in the Graphics to convert the Tangent space normal to world space. 
				2. In comparison with the Rotation matrix, which is put the new rotated axis in the old coordinate basis as the column of the rotation matrix. The result of the rotation is still under the original basis. This can be understood as re-calc the contribution of each scalar along the original basis under the new rotated axises. The change-of-basis and rotation matrix are both SO(3) and mutual inverse. 
			2. Another case is when the new basis and the original basis are expressed of another fundamental basis. Then the change-of-basis is (B_new.inv() * B_ori). And for each B, put the basis on the column.
			
5. Inner Product: 
	1. Orthogonal vector inner product is 0. 
	2. Denote the angle difference of 2 normalized vector. The bigger difference, the smaller value
	3. Use for project 1 vector to another
	4. Determine if the direction is shooting forward the plane normal or into the plane normal (if value > 0)
6. Cross Product (assume right-hand system): 
	1. Self cross product is 0
	2. Determine a point is on the left or right of a vector. Moreover, we can determine if a point is within the triangle. (Rasterization)
	3. Determine the triangle is CCW/CW wind
	4. Use skew-symmetric matrix to transform the cross product calculation to a matrix multiplication. 
	5. Any 2 non-parallel vectors’ cross product can describe a plane’s direction.

7. Group: A closed set of transfer functions. Notice that there is no Commutative Property for the Group. 
	1. General Linear Group GL(n): det(A) != 0
		1. Orthogonal Group O(n): A.inv() = A.transpose() and det(A) = +1/-1. Rank(A) = n.
			1. Special Orthogonal Group SO(n): A is O(n) and det(A) = +1. This is because of right hand system. 
		2. Affine Group A(n): [A, b; 0, 1]
			1. Euclidean Group E(n): T is Affine group also A is orthogonal group. 
				1. Special Euclidean Group SE(n): If A is further a SO(n). Then it is a SE(n)
	2. SO(n) is O(n) is GL(n)
	3. SE(n) is E(n) is A(n) is GL(n+1) (Homogeneous coordinate)
	
8. Gram-Schmidt Process: Given an arbitrary set of basis of a GL(n), we can always transfer them to a set of mutual orthogonal basis. This is done in interation by always deduct the component of projection to the existing orthogonal basis. [Gram–Schmidt process - Wikipedia](https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process)
9. QR Decomposition: Any matrix can be decomposed to A = QR. Gram-Schmidt Process is a kind of QR Decomposition. Q is an orthogonal matrix and R is an upper triangular matrix. QR decomposition can be used for solving least square problem, no matter the A is under-determined or over-determined.  
10. Eigen Decomposition: Eigen vectors are the direction of transform. Eigen vector and Eigen value can be used to decompose a matrix. And also can use for solving inverse, however not recommend to use Eigen Decomposition to solve linear equation.
11. SVD
	1. When matrix is square, then the Eigen value is the singular value. Also det(A) = singular_value[i++] * det(A).
12. Solve a Linear equation Ax = b:
	1. A is non-Singular det(A)!=0:
		1. Calculate A.inv()
		2. Cramer’s rule
	2. A is Singular:
		1. Pseudo-inverse using SVD
		2. LU decomposition
		3. QR decomposition
		4. For Mx = 0, calculate the eigen vector of the (A.t() * A). The solution is the eigen vector corresponding to the smallest eigen value.
		5. For Mx = 0, SVD, the solution is the singular vector corresponding to the smallest singular value. 
		6. For Mx = 0, force the first order partial derivative to be zero. Jacobian matrix to be zero. Might get local minimum.  
		7. Non-linear gradient descent. Might get local minimum.	
	3. [Eigen: Linear algebra and decompositions](https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html)
	4. [Eigen: Solving linear least squares systems](https://eigen.tuxfamily.org/dox/group__LeastSquares.html)

13. Jacobian matrix: first order partial derivative matrix, used as the gradient direction in the non linear optimization. Hessian matrix is the second order partial derivative matrix. 


### Lie Algebra
1. Lie group is the SE(3), Lie Algebra map the Lie Group to the exponential coordinate space, which is more easy to calculate derivative and has meaningful add operation. Just like the Cartesian Coordinate System and the Polar Coordinate System. They both represent the same point but in different format. 
2. Rodrigues formula: Map AngleAxis to Lie Group SO(3). There is another formula using the Trace of matrix to map the SO(3) to the AngleAxis. Moreover we can use AngleAxis to calculate the corresponding mapped value in Lie Algebra: phi=skewsymmetric(w=theta*axis). The direction of the w is the axis and magnitude is the angle. So we now know how to convert between Lie Group and Lie Algebra. (Detail check Slambook chapter 4)
3. Homogeneous Coordinate: Can covert the linear combination to a simple matrix multiplication. 
4. Lie Algebra for SE(3)
5. Later when we talking about the Gradient to the Camera pose [R,t], we are talking about the Jacobian of the camera pose on the Lie Algebra. 
6. P37: Summary


### Image Formation
1. Image through lens. Fundamental equation: P48
2. Image through pinholes. 
3. Intrinsic, The difference between the Projection matrix in Graphics is: Intrinsic K is modeling base on the pinhole camera. K mapped the normalized Camera space position (divided by Z) to pixel coordinate. Projection in graphics map the camera space position to a 3 dimension clipped frustum. Then rescale the frustum to NDC space. 
4. Extrinsic, is the camera pose relative to the world or the previous pose.
5. Radial Distortion coefficient
6. Rendering Function from the Image Sensor perspective: P69

### Image Correspondence
1. Optical Flow: Brightness Constancy Constraint (4.14). A strong assumption that assume the camera or the pixel on the image is moving very slow. So the delta of a pixel’s greyscale is equal to the image gradient at last frame along X and Y axis, times the delta t. Moreover, we can select a m*m window in the image and apply this constraint to each pixel. Then we can form an over-determined linear equation. Solve that can yield the speed of the pixel change. Formula (4.20) is the linear equation for the pixel speed. 
2. Algorithm (4.1): denote an algorithm for compute the pixel displacement with either feature tracking or Optical Flow. The main difference is that 
	1. for feature tracking, the target points are selected corner points (This method also known as the 8-Point algorithm for solving the Epipolar constraint). 
	2. For the Optical Flow, we just need to detect the feature point at the first frame, then use the optical flow to tracking them and solve for the camera movement. This cam save us a lot of computation on Feature detection and matching. (This is also known as LK Optical Flow)

3. Feature Point Detection: Here the book mainly talking about some traditional feature point technique. Morden feature detection include: HOG, SIFT, SURF, ORB and even Deep Learning. 
4. Corner Detection: Algorithm (4.2). Harris Corner Detection is using the EigenValue to characterize the quality of the image gradient matrix. Shi-Tomasi detection: idea is similar to Harris but in different form.

5. Line detector: Canny edge detection (Algorithm 4.3). More over, we can use the line segments (B-splines) to fitting the line base on the image gradient.
6. Sobel Operator: Gradient kernel to convolution with the whole image to determine the image gradient along both X and Y axis for each pixel. 
7. SIFT + FLANN (TODO)


### Epipolar Geometry For Calibrated Views
1. Epipolar Constraint: Equation (5.2). Notice that for essential matrix, the x is the normalized Camera space coordinate [X_Z; Y_Z; 1]. For Fundamental Matrix, the p is the homogeneous pixel coordinate [u; v; 1].
2. Other concepts: 
	1. Epipolar plane
	2. Baseline
	3. Epipoles
	4. Epipolar lines
3. Essential Matrix is E = skew(t) * R. 
4. Fundamental Matrix is F = K.inv().t() * E * K.inv().
5. The eight-point Algorithm: 
	1. Why 8 points? Because of Scale Ambiguity of Monocular SLAM: 
		1. Because the equation to be solved is Mx = 0, which means x times any scalar factor, this equation still stands. So we can always normalize the x to set the last element to be 1. So that we have (9-1) degree of freedom. So only 8 points
		2. Another perspective of this problem is that due to Scale Ambiguity, we can always normalize the norm(x) to be 1. So that we have our 9th equation. We just need 8 more.
	2. Solutions, usually we will have more than 8 pair of matching points:
		1. Linear solution: Since this is an over-determined linear equation, we can use SVD then get smallest singular value or (M.t()*M)’s smallest eigen value to solve it. Smallest because ideally the corresponding Singular value should be 0. 
		2. Non-linear Solution: reform the 8-points equation to become a least-square problem: argmin(x) such as (x.inv() * M.t() * M * x). Then we can use non-linear gradient descent.
		3. Or we can use the RANSAC, which is a cross-validation mechanism. Basically, we random select 8-point pairs and solve for a pose. And for all the possible pose, we calculate which one has least overall reproduction error. Then select that. 
		
5. To recover R and t from E from 2 calibrated views. Assume already obtain the rough estimated E through the 8-points algorithm:
	1. However the E might not on the essential space (manifold). So we need to normalize (retroject) it on the essential space. This is done via SVD and set the diagonal singular matrix to [1, 1, 0] or [(d1+d2)/2, (d1+d2)/2, 0] (d1 and d2 are the two largest singular values). Since this is monocular reconstruction, the world space scale cannot be determine. So it is OK to assign 1 as the normalized singular value. Also notice that for valid essential matrix, the singular value format must be [t, t, 0] (Theorem 5.5). 
	2. To obtain the R and t, we need to re-compose the SVD result of the normalized. We will obtain 4 results and test the triangulated point’s depth must be positive to select the final result. 

6. A complete Epipolar Triangulation (Algorithm 5.1)
	1. Related OpenCV APIs: cv::findEssentialMat, cv::findFundamentalMat, cv::findHomography, cv::recoverPose, cv::triangulatePoints, cv::projectPoints
	2. Using this Monocular SfM we can only can only obtain the normalized camera space coordinate. And will use the first keyframe’s camera space coordinate as the world coordinate. For initialize the Monocular SLAM, we need to have translation between the 2 frame. Then to build a map, ORB-SLAM normalized the median depth of the first keyframe’s feature point as 1 unit. Then we can obtained a normalized translation and other feature points’ depth, then we can starting building the map.
	
7. Planar Scene and Homography:
	1. 4-points Algorithm and [R,t] decomposing: (Algorithm 5.2)
	2. In practice, when most feature points are on the same plane or the camera only have rotation, the Homography matrix works better than fundamental matrix. So usually compute both and select the one that has the lower reprojection error. 
	3. Relationship between Essential Matrix and Homography Matrix: E = skew(t) * H. So when the t is small, the E degenerate to H.
	
8. Continuous Epipolar Geometry for both Essential and Homography: 
	1. Essential Algorithm 5.3
	2. Homography Algorithm 5.4
	
9. Summary: P160



### Epipolar Geometry for uncalibrated views
1. Fundamental Matrix: Pixel coordinate constraint. 
2. Here the camera intrinsic K is unknown. When using Bundle Adjustment, we can add the intrinsic vertex to estimate through graph optimization. 
3. Stratified Reconstruction: with uncalibrated camera, we can only reconstruct up to Projective reconstruction: X_p = H_p * H_a * g_e * X. X is the true structure and g_e is the rigid body motion [R,t]. So the distortion comes from H_p and H_a. 
	1. Projective Reconstruction P188:
		1. After obtain the Fundamental matrix using 8-point algorithm (P212), we can decompose it and use it to solve projective triangulation. 
	2. Affine reconstruction: To obtain a Euclidean Reconstruction, we need to first upgrade the projective reconstruction to Affine Reconstruction
	3. Euclidean Reconstruction
	
### Calibration 
1. Offline Calibration:
	1. With a checkerboard calibration rig/planar
		1. Calibration with Checkerboard P203:
			1. Detect the checkerboard corner points pixel coordinate on image
			2. Define a World coordinate system on the Checkerboard itself. Like the lower left corner is the origin and each square is a unit 1. 
			3. The we have the linear equation:
				1. H = K * [R,t]
				2. cross(x_pixel * H * X_world) = 0
			4. Solve above we get H. Then denote S = K.inv().t() * K.inv(). {h1, h2} as the first and second column of H. Then we have constraint that h1.t() * S * h2 = 0.
			5. Solve for S then we can decompose K
			6. K has 5 DoF (with the skew factor) so we need at least 3 image to get enough linear equation to solve them. 
			
2. Online Calibration:
	1. With partial scene information
		1. Calibrated with vanishing 3 points P199

### Epipolar Summary
1. P206-207


### Step-by-step build of a 3D model from Image using Uncalibrated camera (P375)
1. Feature Selection
2. Feature Matching
	1. Feature tracking
		1. Optical Flow
	2. Brute Force
	3. RANSAC feature matching (P389)
	4. FLANN
	5. Feature buckets
3. Projective reconstruction
	1. Two-view initialization
		1. Obtain Fundamental Matrix with Sampson distance refinement (P393)
	2. Projective Reconstruction (P395)
		1. Decompose F to [R,t] and get 3D structure 
4. Euclidean reconstruction with partial camera knowledge (P401)
	1. Camera Calibration
5. Reconstruction with partial object knowledge
	1. Use prior or semantic information to help reconstruct the 3D model
6. Visualization
	1. Rectify the Epipolar constraint (minimize Sampson distance)
	2. Sparse Feature points to dense mesh triangles
	3. Texture mapping

### Appendix
1. Kalman Filter and EKF (P468)
2. Non-linear optimization (P480)