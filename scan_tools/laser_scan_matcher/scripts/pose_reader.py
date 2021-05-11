#!/usr/bin/env python
import rospy
import time
from std_msgs.msg import Bool
from std_msgs.msg import Float32
from std_msgs.msg import Float64
from ctrl_pkg.msg import ServoCtrlMsg
from geometry_msgs.msg import Pose2D


def set_position(data):
    print("here")



def servo_commands(): 
      
    sub = rospy.Subscriber("/pose2D", Pose2D, set_position)

if __name__ == '__main__':
    try:
       servo_commands()
    except rospy.ROSInterruptException:
        pass
