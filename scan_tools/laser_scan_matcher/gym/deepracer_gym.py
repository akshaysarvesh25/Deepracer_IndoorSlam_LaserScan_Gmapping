#!/usr/bin/env python3
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
gc.collect()
from sac import SAC
from replay_memory import ReplayMemory
from torch.utils.tensorboard import SummaryWriter



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
parser.add_argument('--num_steps', type=int, default=5000000, metavar='N',
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
parser.add_argument('--max_episode_length', type=int, default=3000, metavar='N',
					help='max episode length (default: 3000)')
args = parser.parse_args()

# x_pub = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/output',AckermannDriveStamped,queue_size=1)
x_pub = rospy.Publisher('manual_drive',ServoCtrlMsg,queue_size=1)
pos = [0,0]
yaw_car = 0
MAX_VEL = 0.6
steer_precision = 0 # 1e-3
MAX_STEER = (np.pi*0.25) - steer_precision
MAX_YAW = 2*np.pi
MAX_X = 10
MAX_Y = 10
# target_x = 50/MAX_X
# target_y = 50/MAX_Y
max_lidar_value = 14
# target_point = [target_x,target_y]
THRESHOLD_DISTANCE_2_GOAL = 0.1 #/max(MAX_X,MAX_Y)

class DeepracerGym(gym.Env):

	def __init__(self,target_point):
		super(DeepracerGym,self).__init__()
		
		n_actions = 2 #velocity,steering
		metadata = {'render.modes': ['console']}
		#self.action_space = spaces.Discrete(n_actions)
		self.action_space = spaces.Box(np.array([0., -1.]), np.array([1., 1.]), dtype = np.float32) # speed and steering
		# self.pose_observation_space = spaces.Box(np.array([-1. , -1., -1.]),np.array([1., 1., 1.]),dtype = np.float32)
		# self.lidar_observation_space = spaces.Box(0,1.,shape=(720,),dtype = np.float32)
		# self.observation_space = spaces.Tuple((self.pose_observation_space,self.lidar_observation_space))
		low = np.concatenate((np.array([-1.,-1.,-4.]),np.zeros(720)))
		high = np.concatenate((np.array([1.,1.,4.]),np.zeros(720)))
		self.observation_space = spaces.Box(low,high,dtype=np.float32)
		self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
		self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
		self.reset_simulation_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
		self.target_point_ = np.array([target_point[0]/MAX_X,target_point[1]/MAX_Y])
		#self.lidar_ranges_ = np.zeros(720)
		self.temp_lidar_values_old = np.zeros(720)
	
	def reset(self):        
		global yaw_car
		#time.sleep(1e-2)
		self.stop_car()        
		# rospy.wait_for_service('/gazebo/reset_simulation')
		# try:
		# 	# pause physics
		# 	# reset simulation
		# 	# un-pause physics
		# 	self.pause()
		# 	self.reset_simulation_proxy()
		# 	self.unpause()
		# 	print('Simulation reset')
		# except rospy.ServiceException as exc:
		# 	print("Reset Service did not process request: " + str(exc))

		pose_deepracer = np.array([abs(pos[0]-self.target_point_[0]),abs(pos[1]-self.target_point_[1]), yaw_car],dtype=np.float32) #relative pose 

		# print(len(pose_deepracer), "\n ---------------")
		temp_lidar_values = np.nan_to_num(np.array(lidar_range_values), copy=True, posinf=max_lidar_value)
		# print(len(temp_lidar_values), "\n ---------------")
		temp_lidar_values = temp_lidar_values/max_lidar_value
		temp_lidar_values = np.concatenate((temp_lidar_values, np.zeros(360)))
		return_state = np.concatenate((pose_deepracer,temp_lidar_values))
		# print(return_state.shape, "\n ---------------")
		
		# if ((max(return_state) > 1.) or (min(return_state < -1.)) or (len(return_state) != 723)):
		# 	print('-----------------ERROR Reset----------------------')        
		
		return return_state
	
	def get_reward(self,x,y):
		x_target = self.target_point_[0]
		y_target = self.target_point_[1]
		head = math.atan((self.target_point_[1]-y)/(self.target_point_[0]-x+0.01))
		return -1*(abs(x - x_target) + abs(y - y_target) + abs (head - yaw_car)) # reward is -1*distance to target, limited to [-1,0]

	def step(self,action):
		global yaw_car
		# self.lidar_ranges_ = np.array(lidar_range_values)
		self.temp_lidar_values_old = np.nan_to_num(np.array(lidar_range_values), copy=True, posinf=max_lidar_value)
		self.temp_lidar_values_old = self.temp_lidar_values_old/max_lidar_value

		global x_pub
		msg = ServoCtrlMsg()
		msg.throttle = action[0]*MAX_VEL
		msg.angle = action[1]*MAX_STEER
		x_pub.publish(msg)
		time.sleep(0.03)


		reward = 0
		done = False

		if((abs(pos[0]) < 1.) and (abs(pos[1]) < 1.) ):

			if(min(self.temp_lidar_values_old)<0.4/max_lidar_value):
				reward = -1         
				done = False
				# print('Collission')

			
			elif(abs(pos[0]-self.target_point_[0])<THRESHOLD_DISTANCE_2_GOAL and  abs(pos[1]-self.target_point_[1])<THRESHOLD_DISTANCE_2_GOAL):
				reward = 10            
				done = True
				print('Goal Reached')

			else:
				reward = self.get_reward(pos[0],pos[1])

			pose_deepracer = np.array([abs(pos[0]-self.target_point_[0]),abs(pos[1]-self.target_point_[1]), yaw_car],dtype=np.float32) #relative pose

		else: 
			done = True
			print('Outside Range')
			reward = -1
			temp_pos0 = min(max(pos[0],-1.),1.) #keeping it in [-1.,1.]
			temp_pos1 = min(max(pos[1],-1.),1.) #keeping it in [-1.,1.]

			head = math.atan((self.target_point_[1]-pos[1])/(self.target_point_[0]-pos[0]+0.01)) #calculate pose to target dierction
			pose_deepracer = np.array([abs(pos[0]-self.target_point_[0]),abs(pos[1]-self.target_point_[1]), yaw_car],dtype=np.float32) #relative pose 

		info = {}

		# self.lidar_ranges_ = np.array(lidar_range_values)
		# print(len(pose_deepracer), "\n ---------------")
		temp_lidar_values = np.nan_to_num(np.array(lidar_range_values), copy=True, posinf=max_lidar_value)
		temp_lidar_values = temp_lidar_values/max_lidar_value
		# print(len(temp_lidar_values), "\n ---------------")
		temp_lidar_values = np.concatenate((temp_lidar_values, np.zeros(360)))


		return_state = np.concatenate((pose_deepracer,temp_lidar_values))

		return return_state,reward,done,info     

	def stop_car(self):
		global x_pub
		msg = ServoCtrlMsg()
		msg.throttle = 0.
		msg.angle = 0.
		x_pub.publish(msg)
		time.sleep(0.03)

	def render(self):
		pass

	def close(self):
		pass

lidar_range_values = []
origin = [0,0]
count = 0
DISPLAY_COUNT=10
WB = 0.15 #[m]
DT = 1
velocity = 0
old_pos = [0,0]
delta = 0
T_Horizon = 5
n = 3
m = 2
R = np.diag([0.01, 1.0])  # input cost matrix
Rd = np.diag([0.01, 1.0])  # input difference cost matrix
Q = np.diag([1.0, 1.0, 0.5])  # state cost matrix

class State:
	def __init__(self, x=0.0, y=0.0, yaw=0.0):
		self.x = x
		self.y = y
		self.yaw = yaw
		self.predelta = None





#https://automaticaddison.com/how-to-convert-a-quaternion-into-euler-angles-in-python/
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

def get_vehicle_state(data):
	
	global pos,velocity,old_pos
	pos[0] = data.x/MAX_X
	pos[1] = data.y/MAX_Y
	yaw_car = data.theta
	# racecar_pose = data.pose[2]

def get_current_velocity(x_old,y_old,x_new,y_new):
	vel = math.sqrt(pow((x_old-x_new),2)+pow((x_old-x_new),2))
	return vel

def get_lidar_data(data):
	#print(type(data.ranges))
	# i = 1+1
	global lidar_range_values
	lidar_range_values = np.array(data.ranges,dtype=np.float32)
	# if len(lidar_range_values) < 720:


def start():
	torch.cuda.empty_cache()

	rospy.init_node('deepracer_controller_mpc', anonymous=True)
	
	x = rospy.Subscriber("/pose2D", Pose2D, get_vehicle_state)
	x_sub2 = rospy.Subscriber("/scan", LaserScan, get_lidar_data)


	target_point = [-5, 0]
	env =  DeepracerGym(target_point)

	'''
	while not rospy.is_shutdown():
		time.sleep(1)
		print('---------------------------',check_env(env))
	'''

	# max_time_step = 3000
	# max_eposide = 1
	# e = 0
	# while not rospy.is_shutdown():
	# 	time.sleep(1) #Do not remove this 
	# 	state = env.reset()
	# 	env.stop_car()
	# 	time.sleep(1)        
	# 	while(e < max_eposide):
	# 		e += 1  
	# 		# state = env.reset()          
	# 		for _ in range(max_time_step):
	# 			action = np.array([0.1,-1])
	# 			n_state,reward,done,info = env.step(action)
	# 			# display(n_state[2])
	# 			time.sleep(0.01)
	# 			print(n_state[2],end='\r')
	# 			if done:
	# 				state = env.reset()                   
	# 				break
	# 	return True
	
	# rospy.spin()

	# while not rospy.is_shutdown():
	# 	# Training Script
	# 	rospy.sleep(1) #Do not remove this 
	# 	state = env.reset() #Do not remove this 
	# 	torch.manual_seed(args.seed)
	# 	np.random.seed(args.seed)

	# 	agent = SAC(env.observation_space.shape[0], env.action_space, args)

	# 	#Pretrained Agent
	# 	actor_path = "models/sac_actor_<DeepracerGym instance>_"
	# 	critic_path = "models/sac_critic_<DeepracerGym instance>_"
	# 	agent.load_model(actor_path, critic_path)

	# 	# Memory
	# 	memory = ReplayMemory(args.replay_size, args.seed)
	# 	#Tesnorboard
	# 	writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 'DeepracerGym',
	# 														 args.policy, "autotune" if args.automatic_entropy_tuning else ""))
	# 	total_numsteps = 0
	# 	updates = 0
	# 	num_goal_reached = 0

	# 	for i_episode in itertools.count(1):
	# 		episode_reward = 0
	# 		episode_steps = 0
	# 		done = False
	# 		state = env.reset()
			
	# 		while not done:
	# 			start_time = time.time()
	# 			if args.start_steps > total_numsteps:
	# 				action = env.action_space.sample()  # Sample random action
	# 			else:
	# 				action = agent.select_action(state)  # Sample action from policy

	# 			if len(memory) > args.batch_size:
	# 				# Number of updates per step in environment
	# 				for i in range(args.updates_per_step):
	# 					# Update parameters of all the networks
	# 					critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

	# 					writer.add_scalar('loss/critic_1', critic_1_loss, updates)
	# 					writer.add_scalar('loss/critic_2', critic_2_loss, updates)
	# 					writer.add_scalar('loss/policy', policy_loss, updates)
	# 					writer.add_scalar('loss/entropy_loss', ent_loss, updates)
	# 					writer.add_scalar('entropy_temprature/alpha', alpha, updates)
	# 					updates += 1

	# 			next_state, reward, done, _ = env.step(action) # Step
	# 			# print("Step Time: ",time.time()-start_time,end='\r')
	# 			if (reward > 9) and (episode_steps > 1): #Count the number of times the goal is reached
	# 				num_goal_reached += 1 

	# 			episode_steps += 1
	# 			total_numsteps += 1
	# 			episode_reward += reward
	# 			if episode_steps > args.max_episode_length:
	# 				done = True

	# 			# Ignore the "done" signal if it comes from hitting the time horizon.
	# 			# (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
	# 			mask = 1 if episode_steps == args.max_episode_length else float(not done)
	# 			# mask = float(not done)
	# 			memory.push(state, action, reward, next_state, mask) # Append transition to memory

	# 			state = next_state

	# 		if total_numsteps > args.num_steps:
	# 			break

	# 		if (episode_steps > 1):
	# 			writer.add_scalar('reward/train', episode_reward, i_episode)
	# 			writer.add_scalar('reward/episode_length',episode_steps, i_episode)
	# 			writer.add_scalar('reward/num_goal_reached',num_goal_reached, i_episode)

	# 		print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))
	# 		print("Number of Goals Reached: ",num_goal_reached)

	# 	print('----------------------Training Ending----------------------')
	# 	env.stop_car()

	# 	agent.save_model("corridor_turn", suffix = "1")
	# 	return True

	# rospy.spin()

	# while not rospy.is_shutdown():
	while True:
		# Evaluation script
		# rospy.sleep(1) #Do not remove this 
		state = env.reset() #Do not remove this 
		# torch.manual_seed(args.seed)
		# np.random.seed(args.seed)
		# Trained Agent
		actor_path = "models/sac_actor__DeepracerGym instance__"
		critic_path = "models/sac_critic__DeepracerGym instance__"
		# print(env.observation_space.shape[0])
		agent = SAC(env.observation_space.shape[0], env.action_space, args)
		agent.load_model(actor_path, critic_path)
		print("Model Loaded")
		# Memory
		# memory = ReplayMemory(args.replay_size, args.seed)		
		updates = 0		
		episode_reward = 0
		episode_steps = 0
		done = False
		state = env.reset()
		total_numsteps = 1000000
		num_goal_reached = 0

		while not done and (episode_steps < 3000):
			if args.start_steps > total_numsteps:
				action = env.action_space.sample()  # Sample random action
			else:
				action = agent.select_action(state)  # Sample action from policy
			time.sleep(0.01) # Added delay to make up fo network delay during training

			next_state, reward, done, _ = env.step(action) # Step
			if (reward > 9) and (episode_steps > 1): #Count the number of times the goal is reached
				num_goal_reached += 1 

			episode_steps += 1
			episode_reward += reward

			mask = 1 if episode_steps == args.max_episode_length else float(not done)

			state = next_state

		print('----------------------Evaluation Ending----------------------')
		env.stop_car()

		return True

	rospy.spin()
	

if __name__ == '__main__':
	try:
		Flag = False
		Flag = start()
		if Flag:
			print('----------_All Done-------------')
	except rospy.ROSInterruptException:
		pass
