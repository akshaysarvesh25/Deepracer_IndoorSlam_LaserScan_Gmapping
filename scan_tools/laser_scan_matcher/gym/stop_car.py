#!/usr/bin/env python
import rospy
import time
from std_msgs.msg import Bool
from std_msgs.msg import Float32
from std_msgs.msg import Float64
from ctrl_pkg.msg import ServoCtrlMsg

def servo_commands():
	rospy.init_node('control_node', anonymous=True)
	x_pub = rospy.Publisher('manual_drive',ServoCtrlMsg,queue_size=1)
	msg = ServoCtrlMsg()

	while not rospy.is_shutdown():
		msg.throttle = -0.0
		x_pub.publish(msg)
		time.sleep(2)
		
if __name__ == '__main__':
    try:
        servo_commands()
    except rospy.ROSInterruptException:
        pass
