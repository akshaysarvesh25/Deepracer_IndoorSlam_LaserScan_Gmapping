#!/usr/bin/env python3
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


pos = [0,0]
old_pos = [0,0]
lidar_range_values = np.zeros(360)
yaw_car = 0
MAX_VEL = 1.0
steer_precision = 0 # 1e-3
MAX_STEER = (np.pi/6.0) - steer_precision
MAX_YAW = 2*np.pi
MAX_X = 20
MAX_Y = 20
max_lidar_value = 14
THRESHOLD_DISTANCE_2_GOAL = 0.6/max(MAX_X,MAX_Y)
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



def filtered_data(pose_data,lidar_data):
	print("callback")
	global pos,velocity,old_pos, total_numsteps, done, env, episode_steps, episode_reward, memory, state, ts, x_pub, num_goal_reached, i_episode
	global updates, episode_reward, episode_steps, num_goal_reached, i_episode, max_ep_reward
	pos[0] = pose_data.x/MAX_X
	pos[1] = pose_data.y/MAX_Y	
	yaw = data.theta
	yaw_car = yaw

	global lidar_range_values
	lidar_range_values = np.array(lidar_data.ranges,dtype=np.float32)


	if total_numsteps > args.num_steps:
		print('----------------------Training Ending----------------------')		
		# agent.save_model("corridor_straight", suffix = "2")

	if not done:

		action = agent.select_action(state)  # Sample action from policy	

		# next_state, reward, done, _ = env.step(action) # Step
		rospy.sleep(0.02)

		if (reward > 9) and (episode_steps > 1): #Count the number of times the goal is reached
			num_goal_reached += 1 

		episode_steps += 1
		total_numsteps += 1
		episode_reward += reward

		if episode_steps > args.max_episode_length:
			done = True

		print(episode_steps, end = '\r')
		# Ignore the "done" signal if it comes from hitting the time horizon.
		# (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
		mask = 1 if episode_steps == args.max_episode_length else float(not done)
		# mask = float(not done)
		memory.push(state, action, reward, next_state, mask) # Append transition to memory
		state = next_state
	else:
		# state = env.reset()
		i_episode += 1
		episode_reward = 0
		episode_steps = 0
		done = False

def start():
	global ts
	torch.cuda.empty_cache()	
	rospy.init_node('deepracer_gym', anonymous=True)		
	pose_sub = message_filters.Subscriber("/pose2D", Pose2D)
	lidar_sub = message_filters.Subscriber("/scan", LaserScan)
	ts = message_filters.TimeSynchronizer([pose_sub,lidar_sub],10)
	ts.registerCallback(filtered_data)
	rospy.spin()

if __name__ == '__main__':
	try:
		Flag = False
		Flag = start()
		if Flag:
			print('----------_All Done-------------')
	except rospy.ROSInterruptException:
		pass
