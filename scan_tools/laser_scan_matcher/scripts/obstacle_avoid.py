#!/usr/bin/env python
import rospy
import time
from std_msgs.msg import Bool
from std_msgs.msg import Float32
from std_msgs.msg import Float64
from ctrl_pkg.msg import ServoCtrlMsg
from sensor_msgs.msg import LaserScan
import numpy as np
x_pub = rospy.Publisher('manual_drive',ServoCtrlMsg,queue_size=1)
msg = ServoCtrlMsg()
#higher value means closer obstacle
def get_lidar_data(data):
    print(type(data))
    #print(data.ranges)
    ranges = np.array(data.ranges).astype('float32')
    # Convert data to numpy array
    #print(ranges.size)
    # Check size of array
    ranges_split = [100,100,100,100,100,100,100,100]
    for x in range (0, 8):
        for y in range (0,45):
            if (ranges[45*x+y]>10):
                ranges[45*x+y] = 100
            #ind = (int)((45*x+y-23)/45+1)%8
            #ranges_split[ind] = min(ranges_split[ind],ranges[45*x+y])
            ranges_split[x] = min(ranges_split[x],ranges[45*x+y])
            #print(str(ranges[45*x+y]) + " " + str(ranges_split[x]))
    print(ranges_split)
    # Add condition on lidar vector, to detect obstacle and control vehicle
    # One approach is you can divide the lidar vector into sectors and check
    
    #find out where the sectors begin
    if (ranges_split[7]>1 and ranges_split[0]>1):
        print("moving forward")
        forward()
        stop()
    elif (ranges_split[3]>1 and ranges_split[4]>1):
        print("moving backward")
        reverse()
        stop()
    elif (ranges_split[6]>1 and ranges_split[5]>1):
        print("moving right")
        right()
        stop()
    elif (ranges_split[1]>1 and ranges_split[2]>1):
        print("moving left")
        left()
        stop()
    # whether there is any obstacle in that sector and accordingly take action 
    # to go forward, reverse, left, right

def forward():
    msg.throttle = 0.5
    msg.angle = .05
    x_pub.publish(msg)
    time.sleep(4)
    
def reverse():
    msg.throttle = -0.5
    msg.angle = .05
    x_pub.publish(msg)
    time.sleep(4)
def left():
    msg.throttle = 0.5
    msg.angle = .7 #90 degree turn is 1.571 assuming the 0 radians is in the center of the front
    x_pub.publish(msg)
    time.sleep(4)
def right():
    msg.throttle = 0.5
    msg.angle = -.7 #270 degree turn is 4.712 assuming the 0 radians is in the center of the front (can we use negative angles? where does heading start?)
    x_pub.publish(msg)
    time.sleep(4)
def stop():
    msg.throttle = 0.0
    msg.angle = 0.0
    x_pub.publish(msg)
    time.sleep(5)
def main():
    rospy.init_node('control_node', anonymous=True)    
    lidar_sub = rospy.Subscriber('scan', LaserScan, get_lidar_data)    
    rospy.spin()
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass