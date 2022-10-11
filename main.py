
from cgi import parse_multipart
from email.errors import InvalidMultipartContentTransferEncodingDefect
import imghdr
from os import O_SYNC
from pyexpat.errors import XML_ERROR_INVALID_TOKEN
from re import X
from string import whitespace
from termios import TAB0
from types import FrameType
import click, tqdm, random
from numpy import blackman
from matplotlib.pyplot import gray

from slam import *

def run_dynamics_step(src_dir, log_dir, idx, split, t0=0, draw_fig=False):
    """
    This function is for you to test your dynamics update step. It will create
    two figures after you run it. The first one is the robot location trajectory
    using odometry information obtained form the lidar. The second is the trajectory
    using the PF with a very small dynamics noise. The two figures should look similar.
    """
    slam = slam_t(Q=1e-8*np.eye(3))
    slam.read_data(src_dir, idx, split)

    # trajectory using odometry (xy and yaw) in the lidar data
    d = slam.lidar
    xyth = []
    for p in d:
        xyth.append([p['xyth'][0], p['xyth'][1],p['xyth'][2]])
    xyth = np.array(xyth)

    plt.figure(1); plt.clf();
    plt.title('Trajectory using onboard odometry')
    plt.plot(xyth[:,0], xyth[:,1])
    logging.info('> Saving odometry plot in '+os.path.join(log_dir, 'odometry_%s_%02d.jpg'%(split, idx)))
    plt.savefig(os.path.join(log_dir, 'odometry_%s_%02d.jpg'%(split, idx)))

    # dynamics propagation using particle filter
    # n: number of particles, w: weights, p: particles (3 dimensions, n particles)
    # S covariance of the xyth location
    # particles are initialized at the first xyth given by the lidar
    # for checking in this function
    n = 3
    w = np.ones(n)/float(n)
    p = np.zeros((3,n), dtype=np.float64)
    slam.init_particles(n,p,w)
    slam.p[:,0] = deepcopy(slam.lidar[0]['xyth'])

    print('> Running prediction')
    t0 = 0
    T = len(d)
    ps = deepcopy(slam.p)   # maintains all particles across all time steps
    plt.figure(2); plt.clf();
    ax = plt.subplot(111)
    for t in tqdm.tqdm(range(t0+1,T)):
        slam.dynamics_step(t)
        ps = np.hstack((ps, slam.p))

        if draw_fig:
            ax.clear()
            ax.plot(slam.p[0], slam.p[0], '*r')
            plt.title('Particles %03d'%t)
            plt.draw()
            plt.pause(0.01)

    plt.plot(ps[0], ps[1], '*c')
    plt.title('Trajectory using PF')
    logging.info('> Saving plot in '+os.path.join(log_dir, 'dynamics_only_%s_%02d.jpg'%(split, idx)))
    plt.savefig(os.path.join(log_dir, 'dynamics_only_%s_%02d.jpg'%(split, idx)))

def run_observation_step(src_dir, log_dir, idx, split, is_online=False):
    """
    This function is for you to debug your observation update step
    It will create three particles np.array([[0.2, 2, 3],[0.4, 2, 5],[0.1, 2.7, 4]])
    * Note that the particle array has the shape 3 x num_particles so
    the first particle is at [x=0.2, y=0.4, z=0.1]
    This function will build the first map and update the 3 particles for one time step.
    After running this function, you should get that the weight of the second particle is the largest since it is the closest to the origin [0, 0, 0]
    """
    slam = slam_t(resolution=0.05)
    slam.read_data(src_dir, idx, split)

    # t=0 sets up the map using the yaw of the lidar, do not use yaw for
    # other timestep
    # initialize the particles at the location of the lidar so that we have some
    # occupied cells in the map to calculate the observation update in the next step
    t0 = 0
    xyth = slam.lidar[t0]['xyth']
    xyth[2] = slam.lidar[t0]['rpy'][2]
    logging.debug('> Initializing 1 particle at: {}'.format(xyth))
    slam.init_particles(n=1,p=xyth.reshape((3,1)),w=np.array([1]))

    slam.observation_step(t=0)
    logging.info('> Particles\n: {}'.format(slam.p))
    logging.info('> Weights: {}'.format(slam.w))

    # reinitialize particles, this is the real test
    logging.info('\n')
    n = 3
    w = np.ones(n)/float(n)
    p = np.array([[2, 0.2, 3],[2, 0.4, 5],[2.7, 0.1, 4]])
    slam.init_particles(n, p, w)

    slam.observation_step(t=1)
    logging.info('> Particles\n: {}'.format(slam.p))
    logging.info('> Weights: {}'.format(slam.w))

def run_slam(src_dir, log_dir, idx, split):
    """
    This function runs slam. We will initialize the slam just like the observation_step
    before taking dynamics and observation updates one by one. You should initialize
    the slam with n=100 particles, you will also have to change the dynamics noise to
    be something larger than the very small value we picked in run_dynamics_step function
    above.
    """
    slam = slam_t(resolution=0.05, Q=np.diag([2e-2,2e-2,1e-2]))
    slam.read_data(src_dir, idx, split)
    T = len(slam.lidar)
    # T   =   10000

    # raise NotImplementedError
    # again initialize the map to enable calculation of the observation logp in
    # future steps, this time we want to be more careful and initialize with the
    # correct lidar scan. First find the time t0 around which we have both LiDAR
    # data and joint data
    #### TODO: XXXXXXXXXXX
    # t_init  =   np.argmax(slam.lidar[0]['t']<=slam.joint['t'])
    t0      =   0
    while(slam.lidar[t0]['t'] <= slam.joint['t'][0]):
        t0+=1


    # initialize the occupancy grid using one particle and calling the observation_step
    # function
    #### TODO: XXXXXXXXXXX
    xyth = slam.lidar[t0]['xyth']
    xyth[2] = slam.lidar[t0]['rpy'][2]
    logging.debug('> Initializing 1 particle at: {}'.format(xyth))
    slam.init_particles(n=1,p=xyth.reshape((3,1)),w=np.array([1]))

    slam.observation_step(t=t0)
    logging.info('> Particles\n: {}'.format(slam.p))
    logging.info('> Weights: {}'.format(slam.w))

    # slam, save data to be plotted later
    #### TODO: XXXXXXXXXXX
    # initPose    =   np.random.uniform(np.array([-19.0,-19.0,-np.pi]).reshape((3,1)),np.array([19.0,19.0,np.pi]).reshape((3,1)),(3,100))
    # slam.init_particles(3)#,initPose)
    # n = 3
    # w = np.ones(n)/float(n)
    # p = np.array([[2, 0.2, 3],[2, 0.4, 5],[2.7, 0.1, 4]])
    # slam.init_particles(n, p, w)
    
    slam.init_particles(100)
    
    estimated_trajectory     =   []
    actual_trajectory        =   []
    for t in tqdm.tqdm(range(t0+1,T)):
        slam.dynamics_step(t)
        slam.observation_step(t)
        # p_estimate  =   slam.p[:,np.argmax(slam.w)]
        p_estimate  =   slam.most_probable_particle
        p_map       =   slam.map.grid_cell_from_xy(p_estimate[0],p_estimate[1])
        estimated_trajectory.append(p_map)

        odom_traj             =   slam.lidar[t]['xyth']
        odom_map              =   slam.map.grid_cell_from_xy(odom_traj[0],odom_traj[1])
        actual_trajectory.append(odom_map)

        
        # if (t%200)==0:
        #     occx,occy = np.nonzero(slam.map.cells)
        #     unkx,unky = np.where(slam.map.log_odds==0)
        #     # freex,freey= np.where()
        #     freex,freey = np.where(slam.map.log_odds<0)
        #     plt.plot(unky,unkx,c='0.45')
        #     plt.plot(freey,freex,c='1.0')
        #     plt.plot(occy,occx,'.k',markersize=1)
        #     plt.plot(p_map[1],p_map[0],'.g',markersize=1)
        #     plt.plot(odom_map[1],odom_map[0],'.r',markersize=1)

        #     plt.gca().invert_yaxis()
        #     plt.xlim([0,800])
        #     plt.ylim([0,800])
        #     plt.draw()
        #     plt.pause(0.01)
        # if(t%100==0):
        #     MAP_2_display = 150*np.ones((slam.map.cells.shape[0],slam.map.cells.shape[1],3),dtype=np.uint8)

        #     wall_indices = np.where(slam.map.log_odds > slam.map.log_odds_thresh)
        #     MAP_2_display[occx,occy,:] = [0,0,0]
        #     unexplored_indices = np.where(abs(slam.map.log_odds) < 1e-1)
        #     MAP_2_display[freex,freey,:] = [255,255,255]  
        #     MAP_2_display[occx,occy,:] = [0,0,0]

    # plt.gca().invert_yaxis()
    # plt.show()
    actual_trajectory         = np.array(actual_trajectory)
    estimated_trajectory      = np.array(estimated_trajectory)
 
    # occx,occy = np.nonzero(slam.map.cells)
    occx,occy = np.where(slam.map.log_odds>slam.map.log_odds_thresh)
    # freex,freey = np.nonzero()
    freex,freey =   np.where(slam.map.log_odds<slam.map.log_odds_thresh)
    # slam.map.log_odds[occx,occy]    += 100*slam.lidar_log_odds_occ
    
    # unkx,unky = np.where(abs(slam.map.log_odds)<0.1)
    # plt.plot(unky,unkx,c='0.45')
    # plt.plot(freex,freey,c='1.0')
    # plt.plot(occy,occx,'.k',markersize=1)
    # # plt.plot(p_map[1],p_map[0],'.g',markersize=1)
    # # plt.plot(odom_map[1],odom_map[0],'.r',markersize=1)
    # plt.plot(actual_trajectory[:,1],actual_trajectory[:,0],'.r')
    # plt.plot(estimated_trajectory[:,1],estimated_trajectory[:,0],'.g')
    # plt.xlim([0,800])
    # plt.ylim([0,800])
    # plt.draw()
    # plt.gca().invert_yaxis()
    # # plt.gca().invert_xaxis() 
    # plt.show()
    
    
    map_plot = 255*np.ones((slam.map.cells.shape[0],slam.map.cells.shape[1],3),dtype=np.uint8)
    map_plot[occx,occy,:] = [0,0,0]
    unexplored_indices = np.where(abs(slam.map.log_odds) < 1e-1)
    map_plot[freex,freey,:] = [255,255,255]  
    map_plot[occx,occy,:] = [0,0,0]
    map_plot[unexplored_indices[0],unexplored_indices[1],:] = [150,150,150]
    plt.plot(actual_trajectory[:,1],actual_trajectory[:,0],'.r',markersize=1)
    plt.plot(estimated_trajectory[:,1],estimated_trajectory[:,0],'.g',markersize=1)
    plt.imshow(map_plot)
    plt.show()
    plt.xlim([0,800])
    plt.ylim([0,800])
    plt.draw()
    plt.gca().invert_yaxis()
    plt.pause(0.01)
    plt.show()

    # plt.imshow(slam.map.log_odds)
    # plt.imshow(slam.map.cells)
    # pltMap = 1*(slam.map.log_odds>0)
    
    # img                       =   np.zeros((801,801,3))
    # cellmap = np.array([slam.map.cells,slam.map.cells,slam.map.cells]).T
    # # gray    =   np.argwhere(logOdds<0)
    # black   =   np.argwhere(cellmap==0)
    # white   =   np.argwhere(cellmap<0)
    # # img[gray] = [128,128,128]
    # img[black]       
    # img[estimated_trajectory[:,0],estimated_trajectory[:,1]]  = [0,255,0]
    # img[actual_trajectory[:,0],actual_trajectory[:,1]]        = [255,0,0]
    # plt.imshow(img,cmap="gray",vmin=0, vmax=255)
    # plt.show()



@click.command()
@click.option('--src_dir', default='./', help='data directory', type=str)
@click.option('--log_dir', default='logs', help='directory to save logs', type=str)
@click.option('--idx', default='1', help='dataset number', type=int)
@click.option('--split', default='train', help='train/test split', type=str)
@click.option('--mode', default='slam',
              help='choices: dynamics OR observation OR slam', type=str)
def main(src_dir, log_dir, idx, split, mode):
    # Run python main.py --help to see how to provide command line arguments

    if not mode in ['slam', 'dynamics', 'observation']:
        raise ValueError('Unknown argument --mode %s'%mode)
        sys.exit(1)

    np.random.seed(42)
    random.seed(42)

    if mode == 'dynamics':
        run_dynamics_step(src_dir, log_dir, idx, split)
        sys.exit(0)
    elif mode == 'observation':
        run_observation_step(src_dir, log_dir, idx, split)
        sys.exit(0)
    else:
        p = run_slam(src_dir, log_dir, idx, split)
        return p

if __name__=='__main__':
    main()
