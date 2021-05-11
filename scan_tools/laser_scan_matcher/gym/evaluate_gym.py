#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import rospy
import time
from std_msgs.msg import Bool
from std_msgs.msg import Float32
from std_msgs.msg import Float64
from geometry_msgs.msg import PoseStamped, Pose2D
from ctrl_pkg.msg import ServoCtrlMsg
from sensor_msgs.msg import LaserScan
import numpy as np
import matplotlib.pyplot as plt
import math
import gym
from gym import spaces
from std_srvs.srv import Empty
import argparse
import datetime
import itertools
import torch, gc
import message_filters
gc.collect()
from sac import SAC
from replay_memory import ReplayMemory

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--policy', default="Gaussian",
					help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
					help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
					help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
					help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
					help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
					help='Temperature parameter α determines the relative importance of the entropy\
							term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
					help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
					help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
					help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=500000, metavar='N',
					help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
					help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
					help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
					help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
					help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
					help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda',type=int, default=0, metavar='N',
					help='run on CUDA (default: False)')
parser.add_argument('--max_episode_length', type=int, default=400, metavar='N',
					help='max episode length (default: 3000)')
args = parser.parse_args()
rospy.init_node('deepracer_gym', anonymous=True)
x_pub = rospy.Publisher('manual_drive',ServoCtrlMsg,queue_size=1)

pos = [0,0]
yaw_car = 0
MAX_VEL = 0.6 #1
steer_precision = 0 # 1e-3
MAX_STEER = 4.
MAX_YAW = 2*np.pi
MAX_X = 3
MAX_Y = 3
THRESHOLD_DISTANCE_2_GOAL =  0.5 #0.6/max(MAX_X,MAX_Y)
UPDATE_EVERY = 5
count = 0
total_numsteps = 0
updates = 0
num_goal_reached = 0
done = False
i_episode = 1
episode_reward = 0
max_ep_reward = 0
episode_steps = 0
done = False
# memory = ReplayMemory(args.replay_size, args.seed)

class DeepracerGym(gym.Env):

	def __init__(self):
		super(DeepracerGym,self).__init__()
		
		n_actions = 2 #velocity,steering
		metadata = {'render.modes': ['console']}
		self.action_space = spaces.Box(np.array([0., -1.]), np.array([1., 1.]), dtype = np.float32) # speed and steering
		low = np.array([-1.,-1.,-4.])
		high = np.array([1.,1.,4.])
		self.observation_space = spaces.Box(low,high,dtype=np.float32)
		self.target_point = [0./MAX_X, 0./MAX_Y, 1.57]
		self.pose = [pos[0]/MAX_X, pos[1]/MAX_Y, yaw_car]
		self.action = [0., 0.]
		self.traj_x = [self.pose[0]*MAX_X]
		self.traj_y = [self.pose[1]*MAX_Y]
		self.traj_yaw = [self.pose[2]]

	def reset(self):        
		global yaw_car
		self.stop_car() 
		pose_deepracer = np.array([abs(pos[0]-self.target_point[0]),abs(pos[1]-self.target_point[1]), yaw_car],dtype=np.float32) #relative pose 
		return_state = pose_deepracer      
		
		return return_state

	def step(self,action):
		global yaw_car
		global x_pub
		msg = ServoCtrlMsg()
		msg.throttle = action[0]*MAX_VEL
		msg.angle = action[1]*MAX_STEER
		x_pub.publish(msg)
		time.sleep(0.1)
		reward = 0
		done = False

		if((abs(pos[0]) < 1.) and (abs(pos[1]) < 1.) ):
			if(abs(pos[0]-self.target_point[0])<THRESHOLD_DISTANCE_2_GOAL and abs(pos[1]-self.target_point[1])<THRESHOLD_DISTANCE_2_GOAL):
				reward = 10            
				done = True
				print('Goal Reached')

			# else:
			# 	reward = self.get_reward(pos[0],pos[1])

			pose_deepracer = np.array([abs(pos[0]-self.target_point[0]),abs(pos[1]-self.target_point[1]), yaw_car],dtype=np.float32) #relative pose

		else: 
			done = True
			print('Outside Range')
			reward = -1
			temp_pos0 = min(max(pos[0],-1.),1.) #keeping it in [-1.,1.]
			temp_pos1 = min(max(pos[1],-1.),1.) #keeping it in [-1.,1.]

			head = math.atan((self.target_point[1]-pos[1])/(self.target_point[0]-pos[0]+0.01)) #calculate pose to target dierction
			pose_deepracer = np.array([abs(pos[0]-self.target_point[0]),abs(pos[1]-self.target_point[1]), yaw_car],dtype=np.float32) #relative pose 

		info = {}

		return_state = pose_deepracer
		return return_state,reward,done,info     

	def stop_car(self):
		global x_pub
		msg = ServoCtrlMsg()
		msg.throttle = 0.
		msg.angle = 0.
		x_pub.publish(msg)
		time.sleep(0.2)
	
	def render(self):
		pass

	def close(self):
		pass

env =  DeepracerGym()
actor_path = "models/sac_actor_random_initial_golden1"
critic_path = "models/sac_critic_random_initial_golden1"
agent = SAC(env.observation_space.shape[0], env.action_space, args)
agent.load_model(actor_path, critic_path)
agent = SAC(env.observation_space.shape[0], env.action_space, args)
state = np.zeros(env.observation_space.shape[0])

torch.manual_seed(args.seed)
np.random.seed(args.seed)

def pose_callback(data):

	global pos
	print("Actual State: ", data.x, data.y)
	pos[0] = (data.x + 2.0)/MAX_X  # Add as per offset to go forward
	pos[1] = (data.y - 2.0)/MAX_Y # Subtract as per offset to go right
	yaw_car = data.theta
	state = np.array([pos[0], pos[1], yaw_car])
	print('State',state)	 

	# state = env.reset()
	updates = 0		
	episode_reward = 0
	episode_steps = 0
	total_numsteps = 1000000
	num_goal_reached = 0	

	action = agent.select_action(state)  # Sample action from policy
	# time.sleep(0.02) # Added delay to make up fo network delay during training
	next_state, reward, done, _ = env.step(action) # Step
	time.sleep(0.05)

	if abs(pos[0]) < THRESHOLD_DISTANCE_2_GOAL and abs(pos[1]) < THRESHOLD_DISTANCE_2_GOAL: #Count the number of times the goal is reached

		num_goal_reached += 1 
		done = True

	# state = next_state
	if done:
		print('----------------------Evaluation Ending----------------------')
		env.stop_car()
		x.unregister()

def start():
	global ts
	torch.cuda.empty_cache()		
	x = rospy.Subscriber("/pose2D", Pose2D, pose_callback)
	# state = env.reset()
	rospy.spin()

if __name__ == '__main__':
	try:
		Flag = False
		Flag = start()
		if Flag:
			print('----------_All Done-------------')
	except rospy.ROSInterruptException:
		pass
