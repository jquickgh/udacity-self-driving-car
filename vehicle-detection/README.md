# Part 1: Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[Project Code](https://github.com/jquickgh/CarND-Advanced-Lane-Lines/blob/master/P4_Final.ipynb)  |  [Project Writeup](https://github.com/jquickgh/CarND-Advanced-Lane-Lines/blob/master/README.ipynb)  |  [Foggy Night](https://youtu.be/52CN__qzXDM)  |  [Project Video](https://youtu.be/b2WBX3jGyy4)  |  [Challenge Video](https://youtu.be/W-ZhO3uXfJs) 

Built Computer Vision software pipeline with Color and Perspective Transforms to identify lane boundaries in a video stream.

[//]: # (Image References)

[im02]: ./test1017x_4.jpg "Advanced Lane Finding"

![alt text][im02]

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

# Part 2: Vehicle Detection and Tracking
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[Project Code](https://github.com/jquickgh/CarND-Vehicle-Detection/blob/master/P5_Final.ipynb)  |  [Project Writeup](https://github.com/jquickgh/self-driving-car-engineer-nd/blob/master/p5-vehicle-detection-and-tracking/README.ipynb)  |  [Project Video](https://www.youtube.com/watch?v=7h1iv-9sqys)

Created a vehicle detection and tracking pipeline with OpenCV, histogram of oriented gradients (HOG), and support vector machines (SVM). Optimized and evaluated the model on video data from a automotive camera taken during highway driving.

[//]: # (Image References)

[im01]: ./test1017x_5_large.jpg "Vehicle Detection"

![alt text][im01]

The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Normalize features and randomize selections for training and testing.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run software pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


