from gazebo_msgs.msg import ModelState
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Quaternion

import rospy
import math


class Controller(object):
    def __init__(self):
        rospy.Subscriber('/turtle1/cmd_vel', Twist, self.callback2)
        self.command = 0

    def callback2(self, data):
        if data.linear.x == 2.0:
            self.command = 0
        elif data.linear.x == -2.0:
            self.command = 1
        elif data.angular.z == 2.0:
            self.command = 2
        elif data.angular.z == -2.0:
            self.command = 3
        else:
            self.command = 0


