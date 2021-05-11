#!/usr/bin/env python
import rospy
import time
import math
from std_msgs.msg import Bool
from std_msgs.msg import Float32
from std_msgs.msg import Float64
from ctrl_pkg.msg import ServoCtrlMsg
from geometry_msgs.msg import PoseStamped, Pose2D
#from ackermann_msgs.msg import AckermannDriveStamped
#from gazebo_msgs.msg import ModelStates
import PID_control
import tf
#import pandas as pd 
import numpy as np

# Publish to the Ackermann Control Topic
# For Physical Deepracer #
x_pub = rospy.Publisher('manual_drive',ServoCtrlMsg,queue_size=1)

# For Simulation #
#x_pub = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/output',AckermannDriveStamped,queue_size=1)

# Initiate variables
throttle = 0.0
heading = 0.0
pos=[0,0]
yaw=0.0

# Setup a waypoint counter
global count
count = 0

    
def set_position(data):
    ### Subscribe from topic and parse the pose data for robot ###

    global x_des
    global y_des

    ### To check for single goal point ###
    ### Uncomment these lines and comment lines 114-117 ###
    # x_des = -1
    # y_des = -1

    ### Parsing the data ###

    # For Physical Deepracer #
    pos[0] = data.x
    pos[1] = data.y
    yaw = data.theta

    '''
    # For Simulation #
    racecar_pose = data.pose[-1]   
        
    pos[0] = racecar_pose.position.x
    pos[1] = racecar_pose.position.y
    quaternion = (
            data.pose[-1].orientation.x,
            data.pose[-1].orientation.y,
            data.pose[-1].orientation.z,
            data.pose[-1].orientation.w)
    euler = tf.transformations.euler_from_quaternion(quaternion)
    yaw = euler[2]  
    '''

    ### Previous Error and Heading ###
    prev_err = math.sqrt((x_des-pos[0])**2+(y_des-pos[1])**2)
    print("Error : ", prev_err)
    prev_head = math.atan((y_des-pos[1])/(x_des-pos[0]+0.00001))

    ### PID control of car if error distance < 0.5 else stop the car and wait for next waypoint ###
    # if (prev_err < 0.1) or (abs(pos[0]) > 4) or (abs(pos[1]) > 4):
    if (prev_err < 0.35):
        ("Stopping car")
        stop_car()
        print("Starting next loop")
        # sub.unregister()        
        servo_commands()        
    else:
        t1 = time.time()
        control_car(t1, pos,yaw, prev_err, prev_head)        
        

def control_car(t1,pos,yaw, prev_err, prev_head):
    
    # For Physical Deepracer #
    msg = ServoCtrlMsg()

    # For Simulation #
    # msg = AckermannDriveStamped()

    ### PID for throttle control ###
    speed_control = PID_control.PID(0.5,0.001,0.1)
    err = math.sqrt((x_des-pos[0])**2+(y_des-pos[1])**2)
    dt = time.time() - t1
    throttle = speed_control.Update(dt, prev_err, err)

    ### PID for steer control ###   
    steer_control = PID_control.PID(0.005,0.0,0.001)
    head = math.atan((y_des-pos[1])/(x_des-pos[0]+0.01))
    steer = steer_control.Update(dt, prev_head-yaw, head - yaw)

    steer = 0.3 # min(max(steer, -0.1), 0.8)
    throttle = min(max(throttle, -0.8), 0.8)

    ### Publish the control signals to Ackermann control topic ###
    print ("X : ", pos[0], "Y : ", pos[1])
    print("Throttle : ", round(throttle, 1),"Steer : ",round(steer, 1))
    # For Physical Deepracer #
    
    msg.throttle = throttle
    msg.angle = steer
    time.sleep(0.03)

    # For Simulation #
    '''
    msg.drive.speed = throttle 
    msg.drive.steering_angle = steer
    '''

    x_pub.publish(msg)
    

    # Store previous error
    prev_err = err 

def stop_car():
    print("Goal Reached!") 
    ### Stop the car as is ###
    # For Physical Deepracer #
    msg = ServoCtrlMsg()
    msg.throttle = 0
    msg.angle = 0


    # For Simulation #
    '''
    msg = AckermannDriveStamped()
    msg.drive.speed = 0
    msg.drive.steering_angle = 0
    msg.drive.steering_angle_velocity = 0
    '''

    x_pub.publish(msg)
    

def servo_commands():

    global x_des
    global y_des

    ######### For user-input waypoints ########
    # print("Car is at :",pos[0],pos[1])
    print("Enter Waypoints:")
    x_des = float(input())
    y_des = float(input())    
    
    global sub
    global count

    ### For parsing waypoints from a .csv file ###    
    '''
    x_des = (x[count])
    y_des = (y[count])
    print('Navigating to: ',x_des, y_des)
    count +=1
    '''
     
    #msg = AckermannDriveStamped()

    ### Subscribe to /gazebo/model_states for pose data feedback ###
    # For Physical Deepracer #
    sub = rospy.Subscriber("/pose2D", Pose2D, set_position)

    # For Simulation #
    #sub = rospy.Subscriber("/gazebo/model_states", ModelStates, set_position)    

    while not (rospy.is_shutdown()):
        '''
        '''      

    time.sleep(0.1)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    try:
        # For Physical Deepracer #
        rospy.init_node('control_node', anonymous=True)

        # For Simulation #
        # rospy.init_node('servo_commands', anonymous=True)

        ### For parsing waypoints from a .csv file ###       
        global df
        global x, y
        '''
        df = pd.read_csv('route_smooth.csv')
        x = np.array(df["X"])
        y = np.array(df["Y"])
        '''  

        servo_commands()

    except rospy.ROSInterruptException:
        stop_car()
        sub.unregister()
        pass
