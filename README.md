# SLAM_Unscented_Kalmann_Filter
This is an implementation of https://kodlab.seas.upenn.edu/uploads/Arun/UKFpaper.pdf -- A Quaternion-based Unscented Kalman Filter for
Orientation Tracking. This is not the official repository

# What it does

The `data.mat` files present in the `data/` directory contains data from a Planar LIDAR sensor and the robot's IMU sensor. The algorithm reads the data, gets the sequence of sensor inputs that are temporally consistent and implements an Unscented Kalmann Filter to so Simulataneous Localization and Tracking

# Results

The robot's ground truth directory is given in red, and the predicted localization trajectory is given in green. The occupancy map is built on the fly simultaneously with localization




