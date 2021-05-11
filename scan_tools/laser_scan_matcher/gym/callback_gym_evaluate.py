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
import random

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
parser.add_argument('--batch_size', type=int, default=512, metavar='N',
					help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=500000, metavar='N',
					help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
					help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
					help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=1, metavar='N',
					help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
					help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=100000, metavar='N',
					help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda',type=int, default=1, metavar='N',
					help='run on CUDA (default: False)')
parser.add_argument('--max_episode_length', type=int, default=200, metavar='N',
					help='max episode length (default: 3000)')
args = parser.parse_args()
rospy.init_node('deepracer_gym', anonymous=True)
x_pub = rospy.Publisher('manual_drive',ServoCtrlMsg,queue_size=1)

pos = [0,0]
old_pos = [0,0]
lidar_range_values = np.zeros(360)
yaw_car = 0
MAX_VEL = 0.8
steer_precision = 0 # 1e-3
MAX_STEER = np.pi
MAX_YAW = 2*np.pi
MAX_X = 10
MAX_Y = 10
max_lidar_value = 14
THRESHOLD_DISTANCE_2_GOAL = 0.2/max(MAX_X,MAX_Y)
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
memory = ReplayMemory(args.replay_size, args.seed)

class DeepracerGym(gym.Env):

	def __init__(self):
		super(DeepracerGym,self).__init__()
		
		n_actions = 2 #velocity,steering
		metadata = {'render.modes': ['console']}
		#self.action_space = spaces.Discrete(n_actions)
		self.action_space = spaces.Box(np.array([0., -1.]), np.array([1., 1.]), dtype = np.float32) # speed and steering
		# self.pose_observation_space = spaces.Box(np.array([-1. , -1., -1.]),np.array([1., 1., 1.]),dtype = np.float32)
		# self.lidar_observation_space = spaces.Box(0,1.,shape=(720,),dtype = np.float32)
		# self.observation_space = spaces.Tuple((self.pose_observation_space,self.lidar_observation_space))
		low = np.array([-1.,-1.,-4.])
		high = np.array([1.,1.,4.])
		self.observation_space = spaces.Box(low,high,dtype=np.float32)
		self.target_point_ = np.array([0./MAX_X, 0./MAX_Y])
		self.pose_deepracer = np.zeros(3)
		#self.lidar_ranges_ = np.zeros(720)
		# self.temp_lidar_values_old = np.zeros(8)
	
	def reset(self):        
		global yaw_car, lidar_range_values
		#time.sleep(1e-2)
		self.stop_car()        
		head_to_target = self.get_heading(self.pose_deepracer, self.target_point_)
		self.pose_deepracer = np.array([abs(pos[0]-self.target_point_[0]),abs(pos[1]-self.target_point_[1]), yaw_car - head_to_target], dtype=np.float32) #relative pose 
		# temp_lidar_values = np.nan_to_num(np.array(lidar_range_values), copy=True, posinf=max_lidar_value)
		# temp_lidar_values = temp_lidar_values/max_lidar_value
		# temp_lidar_values = np.min(temp_lidar_values.reshape(-1,45), axis = 1)
		# return_state = np.concatenate((self.pose_deepracer,temp_lidar_values))

		# random_targets = [[1., -1.], [1., 0.], [1., 1.]]
		# target_point = random.choice(random_targets)
		# self.target_point_ = np.array([target_point[0]/MAX_X,target_point[1]/MAX_Y])
		print("Episode Target Point : ", self.target_point_)
		
		# if ((max(return_state) > 1.) or (min(return_state < -1.)) or (len(return_state) != 723)):
		# 	print('-----------------ERROR Reset----------------------')        
		
		return self.pose_deepracer

	def get_distance(self,x1,x2):
		# Distance between points x1 and x2
		return math.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)

	def get_heading(self, x1,x2):
		# Heading between points x1,x2 with +X axis
		return math.atan2((x2[1] - x1[1]), (x2[0] - x1[0]))	
	
	def get_reward(self,x,y):
		x_target = self.target_point_[0]
		y_target = self.target_point_[1]
		head_to_target = self.get_heading(self.pose_deepracer, self.target_point_)
		self.pose_deepracer = np.array([abs(pos[0]-self.target_point_[0]),abs(pos[1]-self.target_point_[1]), yaw_car - head_to_target],dtype=np.float32)
		# x = self.pose_deepracer[0]
		# y = self.pose_deepracer[1]
		alpha = head_to_target - yaw_car
		ld = self.get_distance(self.pose_deepracer, self.target_point_)
		crossTrackError = math.sin(alpha) *ld
		return -1*(abs(crossTrackError)**2 + abs(x - x_target) + abs(y - y_target) + 3*abs (yaw_car - head_to_target)/1.57)/6 # reward is -1*distance to target, limited to [-1,0]
	
	def get_reward1(self,x,y):
		x_target = self.target_point_[0]
		y_target = self.target_point_[1]
		head = math.atan((self.target_point_[1]-y)/(self.target_point_[0]-x+0.01))
		return -(1/3)*(abs(x - x_target) + abs(y - y_target) + abs ((1/3.14)*(head - yaw_car))) # reward is -1*distance to target, limited to [-1,0]

	def step(self,action):
		global yaw_car, lidar_range_values
		# self.lidar_ranges_ = np.array(lidar_range_values)
		self.temp_lidar_values_old = np.nan_to_num(np.array(lidar_range_values), copy=True, posinf=max_lidar_value)
		self.temp_lidar_values_old = self.temp_lidar_values_old/max_lidar_value
		self.temp_lidar_values_old = np.min(self.temp_lidar_values_old.reshape(-1,45), axis = 1)
		# print("Least distance to obstacle: ", min(self.temp_lidar_values_old), end = '\r')

		global x_pub
		msg = ServoCtrlMsg()
		msg.throttle = action[0]*MAX_VEL 
		msg.angle = -action[1]*MAX_STEER
		x_pub.publish(msg)
		time.sleep(0.05)

		reward = 0
		done = False

		if((abs(pos[0]) < 1.) and (abs(pos[1]) < 1.) ):

			if(abs(pos[0]-self.target_point_[0])<THRESHOLD_DISTANCE_2_GOAL and abs(pos[1]-self.target_point_[1])<THRESHOLD_DISTANCE_2_GOAL):
				reward = 10            
				done = True
				print('Goal Reached : ', self.target_point_)
				self.stop_car()
				# print("x distance: {:.2f}, y distance : {:.2f}".format(abs(pos[0]-self.target_point_[0]), abs(pos[1]-self.target_point_[1])))

			else:
				reward = self.get_reward(pos[0],pos[1])

			head_to_target = self.get_heading(self.pose_deepracer, self.target_point_)
			self.pose_deepracer = np.array([abs(pos[0]-self.target_point_[0]),abs(pos[1]-self.target_point_[1]), yaw_car - head_to_target],dtype=np.float32) #relative pose

		else: 
			done = True
			print('Outside Range')
			self.stop_car()
			reward = -1
			temp_pos0 = min(max(pos[0],-1.),1.) #keeping it in [-1.,1.]
			temp_pos1 = min(max(pos[1],-1.),1.) #keeping it in [-1.,1.]
			print("x distance: {:.2f}, y distance : {:.2f}".format(abs(pos[0]-self.target_point_[0]), abs(pos[1]-self.target_point_[1])))
			head_to_target = self.get_heading(self.pose_deepracer, self.target_point_)
			self.pose_deepracer = np.array([abs(pos[0]-self.target_point_[0]),abs(pos[1]-self.target_point_[1]), yaw_car - head_to_target],dtype=np.float32) #relative pose 

		info = {}

		# self.lidar_ranges_ = np.array(lidar_range_values)
		
		# temp_lidar_values = np.nan_to_num(np.array(lidar_range_values), copy=True, posinf=max_lidar_value)
		# temp_lidar_values = temp_lidar_values/max_lidar_value
		# temp_lidar_values = np.min(temp_lidar_values.reshape(-1,45), axis = 1)

		# return_state = np.concatenate((self.pose_deepracer, temp_lidar_values))
		# print("Reward : ", reward, end = '\r')
		
		# if ((max(return_state) > 1.) or (min(return_state < -1.)) or (len(return_state) != 723)):
		# 	print('-----------------ERROR Step----------------------')
		# 	print(max(self.pose_deepracer),max(temp_lidar_values))
		# 	print(min(pose_deepracer),min(temp_lidar_values))
		# 	print(len(return_state))
		# 	print('-------------------------------------------------')

		return self.pose_deepracer,reward,done,info     

	def stop_car(self):
		global x_pub
		msg = ServoCtrlMsg()
		msg.throttle = 0.
		msg.angle = 0.
		x_pub.publish(msg)
		time.sleep(1)
	
	def render(self):
		pass

	def close(self):
		pass

env =  DeepracerGym()
actor_path = "models/sac_actor_random_initial_golden1"
critic_path = "models/sac_critic_random_initial_golden1"
agent = SAC(env.observation_space.shape[0], env.action_space, args) 
agent.load_model(actor_path, critic_path) 

torch.manual_seed(args.seed)
np.random.seed(args.seed)

def euler_from_quaternion(x, y, z, w):
	"""
	Convert a quaternion into euler angles (roll, pitch, yaw)
	roll is rotation around x in radians (counterclockwise)
	pitch is rotation around y in radians (counterclockwise)
	yaw is rotation around z in radians (counterclockwise)
	"""
	t0 = +2.0 * (w * x + y * z)
	t1 = +1.0 - 2.0 * (x * x + y * y)
	roll_x = math.atan2(t0, t1)

	t2 = +2.0 * (w * y - z * x)
	t2 = +1.0 if t2 > +1.0 else t2
	t2 = -1.0 if t2 < -1.0 else t2
	pitch_y = math.asin(t2)

	t3 = +2.0 * (w * z + x * y)
	t4 = +1.0 - 2.0 * (y * y + z * z)
	yaw_z = math.atan2(t3, t4)

	return roll_x, pitch_y, yaw_z # in radians

# def lidar_callback(lidar_data):
# 	global lidar_range_values
# 	lidar_range_values = np.array(lidar_data.ranges,dtype=np.float32)

def pose_callback(data):
	global pos,velocity,old_pos, total_numsteps, done, env, episode_steps, episode_reward, memory, state, ts, x_pub, num_goal_reached, i_episode
	global updates, episode_reward, episode_steps, num_goal_reached, i_episode, max_ep_reward, lidar_range_values
	# print("Actual State: ", data.x, data.y)
	pos[0] = (data.x + 2.0)/MAX_X  # Add as per offset to go forward
	pos[1] = (data.y + 2.0 )/MAX_Y # Subtract as per offset to go right
	yaw_car = data.theta
	done = False

	head_to_target = env.get_heading([pos[0], pos[1]], [0., 0.])
	state = np.array([abs(pos[0]),abs(pos[1]), yaw_car - head_to_target],dtype=np.float32)
	# temp_lidar_values = np.nan_to_num(np.array(lidar_range_values), copy=True, posinf=max_lidar_value)
	# temp_lidar_values = temp_lidar_values/max_lidar_value
	# temp_lidar_values = np.min(temp_lidar_values.reshape(-1,45), axis = 1)
	# state = np.concatenate((pose,temp_lidar_values)) #np.array([pos[0], pos[1], yaw_car])

	# print("State :", state)
	action = agent.select_action(state, evaluate = True)  # Sample action from policy	

	# next_state, reward, done, _ = env.step(action) # Step
	print("State : ", state)
	print("Action : ", action)
	# time.sleep(0.1)

	episode_steps += 1
	total_numsteps += 1
	episode_reward += reward

	# print("x distance: {:.2f}, y distance : {:.2f}".format(abs(pos[0]-env.target_point_[0]), abs(pos[1]-env.target_point_[1])), end = '\r')
	if done:
		env.stop_car()
		print("Goal reached {} times out of {} total episodes". format(num_goal_reached, i_episode))
		x.unregister()
		i_episode += 1				
		episode_reward = 0
		episode_steps = 0
		done = False

def start():
	global ts, state
	torch.cuda.empty_cache()	
	rospy.init_node('deepracer_gym', anonymous=True)
	# lidar_sub = rospy.Subscriber("/scan", LaserScan, lidar_callback)		
	x = rospy.Subscriber("/pose2D", Pose2D, pose_callback)
	
	# lidar_sub = message_filters.Subscriber("/scan", LaserScan)
	# ts = message_filters.ApproximateTimeSynchronizer([pose_sub,lidar_sub],10,0.1,allow_headerless=True)
	# ts.registerCallback(filtered_data)
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

