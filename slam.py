
from calendar import c
from cmath import log
from ctypes.wintypes import HACCEL
from http.client import PARTIAL_CONTENT
import os, sys, pickle, math
from re import I, U
from ossaudiodev import control_labels
from pyexpat.errors import XML_ERROR_ENTITY_DECLARED_IN_PE
from copy import deepcopy
from this import d
from tkinter import N
from urllib.request import proxy_bypass

from scipy import io
import numpy as np
import matplotlib.pyplot as plt

from load_data import load_lidar_data, load_joint_data, joint_name_to_index
from utils import *

import logging
logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))

class map_t:
    """
    This will maintain the occupancy grid and log_odds. You do not need to change anything
    in the initialization
    """
    def __init__(s, resolution=0.05):
        s.resolution = resolution
        s.xmin, s.xmax = -20, 20
        s.ymin, s.ymax = -20, 20
        s.szx = int(np.ceil((s.xmax-s.xmin)/s.resolution+1))
        s.szy = int(np.ceil((s.ymax-s.ymin)/s.resolution+1))

        # binarized map and log-odds
        s.cells = np.zeros((s.szx, s.szy), dtype=np.int8)
        s.log_odds = np.zeros(s.cells.shape, dtype=np.float64)
        s.free_cells = np.zeros(s.cells.shape,dtype=np.int8)

        # value above which we are not going to increase the log-odds
        # similarly we will not decrease log-odds of a cell below -max
        s.log_odds_max = 5e6
        # number of observations received yet for each cell
        s.num_obs_per_cell = np.zeros(s.cells.shape, dtype=np.uint64)

        # we call a cell occupied if the probability of
        # occupancy P(m_i | ... ) is >= occupied_prob_thresh
        s.occupied_prob_thresh = 0.6
        s.log_odds_thresh = np.log(s.occupied_prob_thresh/(1-s.occupied_prob_thresh))

    def grid_cell_from_xy(s, x, y):
        """
        x and y are 1-dimensional arrays, compute the cell indices in the map corresponding
        to these (x,y) locations. You should return an array of shape 2 x len(x). Be
        careful to handle instances when x/y go outside the map bounds, you can use
        np.clip to handle these situations.
        """
        #### TODO: XXXXXXXXXXX
        # x       =   np.clip(x,s.xmin,s.xmax)
        # y       =   np.clip(y,s.ymin,s.ymax)
        x_idx   =   (x-   s.xmin)//s.resolution 
        y_idx   =   (y-   s.ymin)//s.resolution 
        x_idx   =   np.clip(x_idx,0,s.szx-1)
        y_idx   =   np.clip(y_idx,0,s.szy-1)

        grid_idx    =   np.array([x_idx,y_idx]).astype(int)
        return grid_idx
        # raise NotImplementedError

class slam_t:
    """
    s is the same as self. In Python it does not really matter
    what we call self, s is shorter. As a general comment, (I believe)
    you will have fewer bugs while writing scientific code if you
    use the same/similar variable names as those in the mathematical equations.
    """
    def __init__(s, resolution=0.05, Q=1e-3*np.eye(3),
                 resampling_threshold=0.3):
        s.init_sensor_model()

        # dynamics noise for the state (x,y,yaw)
        s.Q = Q
        
        # we resample particles if the effective number of particles
        # falls below s.resampling_threshold*num_particles
        s.resampling_threshold = resampling_threshold

        # initialize the map
        s.map = map_t(resolution)

    def read_data(s, src_dir, idx=0, split='train'):
        """
        src_dir: location of the "data" directory
        """
        logging.info('> Reading data')
        s.idx = idx
        s.lidar = load_lidar_data(os.path.join(src_dir,
                                               'data/%s/%s_lidar%d'%(split,split,idx)))
        s.joint = load_joint_data(os.path.join(src_dir,
                                               'data/%s/%s_joint%d'%(split,split,idx)))

        # finds the closets idx in the joint timestamp array such that the timestamp
        # at that idx is t

        s.find_joint_t_idx_from_lidar = lambda t: np.argmin(np.abs(s.joint['t']-t))
        # s.find_joint_t_idx_from_lidar = lambda t: np.argmax(s.lidar[t]['t'] <=s.joint['t'])

    def init_sensor_model(s):
        # lidar height from the ground in meters
        s.head_height = 0.93 + 0.33
        s.lidar_height = 0.15

        # dmin is the minimum reading of the LiDAR, dmax is the maximum reading
        s.lidar_dmin = 1e-3
        s.lidar_dmax = 30
        s.lidar_angular_resolution = 0.25
        # these are the angles of the rays of the Hokuyo
        s.lidar_angles = np.arange(-135,135+s.lidar_angular_resolution,
                                   s.lidar_angular_resolution)*np.pi/180.0

        # sensor model lidar_log_odds_occ is the value by which we would increase the log_odds
        # for occupied cells. lidar_log_odds_free is the value by which we should decrease the
        # log_odds for free cells (which are all cells that are not occupied)
        s.lidar_log_odds_occ = np.log(9)
        s.lidar_log_odds_free = np.log(1/9.)

    def init_particles(s, n=100, p=None, w=None, t0=0):
        """
        n: number of particles
        p: xy yaw locations of particles (3xn array)
        w: weights (array of length n)
        """
        s.n = n
        s.p = deepcopy(p) if p is not None else np.zeros((3,s.n), dtype=np.float64)
        s.w = deepcopy(w) if w is not None else np.ones(n)/float(s.n)
        
        s.most_probable_particle = s.p[:,0]

    @staticmethod
    def stratified_resampling(p, w):
        """
        resampling step of the particle filter, takes p = 3 x n array of
        particles with w = 1 x n array of weights and returns new particle
        locations (number of particles n remains the same) and their weights
        """

        #### TODO: XXXXXXXXXXX
        n               =   w.shape[0]
        r               =   np.random.uniform(0,1.0/n)
        new_particles   =   [] 
        i               =   0 
        c               =   w[0]
        for m in range(n):
            u           =   r + (m/n)
            # c           =   c+w[0]
            while(u>c):
                i       =   i+1
                c       =   c+w[i]
            new_particles.append(p[:,i])
        new_particles   =   np.array(new_particles).T
        return new_particles,(1.0/n)*np.ones(w.shape)                
        # raise NotImplementedError

    @staticmethod
    def log_sum_exp(w):
        return w.max() + np.log(np.exp(w-w.max()).sum())

    def rays2world(s, p, d, head_angle=0, neck_angle=0, angles=None):
        """
        p is the pose of the particle (x,y,yaw)
        angles = angle of each ray in the body frame (this will usually
        be simply s.lidar_angles for the different lidar rays)
        d = is an array that stores the distance of along the ray of the lidar, for each ray (the length of d has to be equal to that of angles, this is s.lidar[t]['scan'])
        Return an array 2 x num_rays which are the (x,y) locations of the end point of each ray
        in world coordinates
        """
        #### TODO: XXXXXXXXXXX
        # raise NotImplementedError

        # make sure each distance >= dmin and <= dmax, otherwise something is wrong in reading
        # the data
        idx                =    np.where((d<=s.lidar_dmax) & (d>=s.lidar_dmin))
        angles             =    angles[idx]
        d                  =    d[idx]

        # 1. from lidar distances to points in the LiDAR frame
        xLidar             =   d*np.cos(angles)
        yLidar             =   d*np.sin(angles)
        xy_lidar           =   np.array([xLidar,yLidar])
        xyz_lidar = np.vstack((xy_lidar,np.zeros(xy_lidar.shape[1])))
        xyz_lidar_hom       =   make_homogeneous_coords_3d(xyz_lidar)   

        # 2. from LiDAR frame to the body frame
        T_lidar_body       =   np.array([0,0,s.lidar_height]).reshape((3,-1))
        H_lidar_body       =   euler_to_se3(0,head_angle,neck_angle,T_lidar_body)
        xyz_body_hom        =   H_lidar_body@xyz_lidar_hom

        # 3. from body frame to world frame
        T_body_world       =   np.array([p[0],p[1],s.head_height]).reshape((3,-1))
        H_body_world       =   euler_to_se3(0,0,p[-1],T_body_world)
        xyz_world_hom      =   H_body_world@xyz_body_hom

        xy_world               =    xyz_world_hom[:-2,:]

        return xy_world

    def get_control(s, t):
        """
        Use the pose at time t and t-1 to calculate what control the robot could have taken
        at time t-1 at state (x,y,th)_{t-1} to come to the current state (x,y,th)_t. We will
        assume that this is the same control that the robot will take in the function dynamics_step
        below at time t, to go to time t-1. need to use the smart_minus_2d function to get the difference of the two poses and we will simply set this to be the control (delta x, delta y, delta theta)
        """

        if t == 0:
            return np.zeros(3)

        #### TODO: XXXXXXXXXXX
        # raise NotImplementedError
        p2  =   s.lidar[t]['xyth']
        p1  =   s.lidar[t-1]['xyth']

        control =   smart_minus_2d(p2,p1)

        return control

    def dynamics_step(s, t):
        """"
        Compute the control using get_control and perform that control on each particle to get the updated locations of the particles in the particle filter, remember to add noise using the smart_plus_2d function to each particle
        """
        #### TODO: XXXXXXXXXXX
        # raise NotImplementedError
        control                 =   s.get_control(t)
        for i in range(s.p.shape[1]):
            p1                      =   s.p[:,i]
            updated_particle        =   smart_plus_2d(p1,control)
            updated_particle_noisy  =   smart_plus_2d(updated_particle,np.random.multivariate_normal([0,0,0],s.Q))  
            s.p[:,i]                =   updated_particle_noisy
        # return updated_particle_noisy

    @staticmethod
    def update_weights(w, obs_logp):
        """
        Given the observation log-probability and the weights of particles w, calculate the
        new weights as discussed in the writeup. Make sure that the new weights are normalized
        """
        #### TODO: XXXXXXXXXXX
        v               = np.log(w) +obs_logp
        p               = v - slam_t.log_sum_exp(v)
        updated_weights           = np.exp(p)
        # updated_weights =   np.multiply(w,prob)
        # if(np.sum(updated_weights)!=0):
        #     updated_weights /=  np.sum(updated_weights)
        return updated_weights
        # raise NotImplementedError

    def observation_step(s, t):
        """
        This function does the following things
            1. updates the particles using the LiDAR observations
            2. updates map.log_odds and map.cells using occupied cells as shown by the LiDAR data

        Some notes about how to implement this.
            1. As mentioned in the writeup, for each particle
                (a) First find the head, neck angle at t (this is the same for every particle)
                (b) Project lidar scan into the world frame (different for different particles)
                (c) Calculate which cells are obstacles according to this particle for this scan,
                calculate the observation log-probability
            2. Update the particle weights using observation log-probability
            3. Find the particle with the largest weight, and use its occupied cells to update the map.log_odds and map.cells.
        You should ensure that map.cells is recalculated at each iteration (it is simply the binarized version of log_odds). map.log_odds is of course maintained across iterations.
        """
        #### TODO: XXXXXXXXXXX
        d           = s.lidar[t]['scan']
        head_angle  = s.joint['head_angles'][joint_name_to_index['Head'],s.find_joint_t_idx_from_lidar(s.lidar[t]['t'])]
        neck_angle  = s.joint['head_angles'][joint_name_to_index['Neck'],s.find_joint_t_idx_from_lidar(s.lidar[t]['t'])]
        angles      = s.lidar_angles
    
        if(t==0):
            most_probable_particle  =   s.p[:,0]

        else:    
            log_probs   =   []
            for j in range(s.p.shape[1]):
                particle                                        =   s.p[:,j]
                xy_world_obs                                    =   s.rays2world(particle,d,head_angle,neck_angle,angles)
                map_idxs                                        =   s.map.grid_cell_from_xy(xy_world_obs[0],xy_world_obs[1])
                s.map.num_obs_per_cell[map_idxs[0],map_idxs[1]] +=  1
                occupied_lidar_particle                         =   s.map.cells[map_idxs[0],map_idxs[1]]
                obs_log_prob                                    =   np.sum(occupied_lidar_particle)
                log_probs.append(obs_log_prob)
            log_probs   =   np.array(log_probs)
            s.w =   s.update_weights(s.w,log_probs)
            most_probable_particle                  =   s.p[:,np.argmax(s.w)]

            
        s.most_probable_particle                =   most_probable_particle
        s.map.free_cells                        =   s.map.free_cells  
        xy_world_obs                            =   s.rays2world(most_probable_particle,d,head_angle,neck_angle,angles)
        
        
        
        map_idxs                                =   s.map.grid_cell_from_xy(xy_world_obs[0],xy_world_obs[1])
        
        x_obs                                   =   map_idxs[0]
        y_obs                                   =   map_idxs[1]
        
        most_probable_particle_grid             =   s.map.grid_cell_from_xy(most_probable_particle[0],most_probable_particle[1])
        particle_x                              =   most_probable_particle_grid[0]
        particle_y                              =   most_probable_particle_grid[1]
        
        x_f = np.ndarray.flatten(np.linspace([particle_x]*len(x_obs), x_obs-1, endpoint=False).astype(int))
        y_f = np.ndarray.flatten(np.linspace([particle_y]*len(x_obs), y_obs-1, endpoint=False).astype(int))

        s.map.free_cells[x_f,y_f]               =   1
        s.map.log_odds[map_idxs[0],map_idxs[1]] +=  s.lidar_log_odds_occ
        s.map.log_odds[x_f,y_f]                 +=  s.lidar_log_odds_free*0.3   
        # s.map.log_odds                          +=  s.lidar_log_odds_free*0.2
        s.map.log_odds                          =   np.clip(s.map.log_odds,-s.map.log_odds_max,s.map.log_odds_max)
        
    
        s.map.cells[s.map.log_odds>=s.map.log_odds_thresh]         =   1   #Occupied -> Map update
        s.map.cells[s.map.log_odds<s.map.log_odds_thresh]          =   0
        if(np.max(s.map.cells)==0):
            flag = True

        s.resample_particles()


        # raise NotImplementedError

    def resample_particles(s):
        """
        Resampling is a (necessary) but problematic step which introduces a lot of variance
        in the particles. We should resample only if the effective number of particles
        falls below a certain threshold (resampling_threshold). A good heuristic to
        calculate the effective particles is 1/(sum_i w_i^2) where w_i are the weights
        of the particles, if this number of close to n, then all particles have about
        equal weights and we do not need to resample
        """
        e = 1/np.sum(s.w**2)
        logging.debug('> Effective number of particles: {}'.format(e))
        if e/s.n < s.resampling_threshold:
            s.p, s.w = s.stratified_resampling(s.p, s.w)
            logging.debug('> Resampling')
