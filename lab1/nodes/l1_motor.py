#!/usr/bin/env python3
import rospy
import math
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import time


def publisher_node():
    """TODO: initialize the publisher node here, \
            and publish wheel command to the cmd_vel topic')"""
    cmd_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
    rate = rospy.Rate(100)

    twist = Twist()
    twist.linear.x = 0.25
    twist.angular.z = 0
    # cmd_pub.publish(twist)


    for _ in range(400): # 10 seconds at 10 Hz 
        cmd_pub.publish(twist)
        rate.sleep()
        
    
    twist.linear.x = 0
    twist.angular.z = 0.5  

    for _ in range(1200):
        cmd_pub.publish(twist)
        rate.sleep() 

    cmd_pub.publish(Twist())

def main():
    try:
        rospy.init_node('motor')
        publisher_node()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()

