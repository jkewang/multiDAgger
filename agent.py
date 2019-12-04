import rospy
import math
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Quaternion


class Agent(object):
    def __init__(self, name):
        self.name = "test_dagger_robot_" + name
        self.observe = [30 for i in range(180)]
        self.target = [0, 0]
        self.x = 0
        self.y = 0
        self.heading = 0
        self.del_heading = 0

        self.v = 0
        self.yaw = 0

        if name == "red":
            rospy.Subscriber("/hokuyo_laser_red", LaserScan, self.callback)
        elif name == "green":
            rospy.Subscriber("/hokuyo_laser_green", LaserScan, self.callback)
        else:
            pass

        rospy.Subscriber("/gazebo/model_states", ModelStates, self.callback2)
        self.pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10)

    def reset(self):
        self.observe = [30 for i in range(180)]
        self.target = [0, 0]
        self.x = 0
        self.y = 0
        self.heading = 0

        self.v = 0
        self.yaw = 0

    def ask_state(self):
        del_d = math.sqrt((self.target[0] - self.x)**2 + (self.target[1] - self.y)**2)
        del_heading = self.heading - math.atan2(self.target[1] - self.y, self.target[0] - self.x)
        if del_heading > math.pi:
            del_heading -= 2 * math.pi
        elif del_heading < -math.pi:
            del_heading += 2 * math.pi
        else:
            pass

        self.del_heading = del_heading
        return self.observe + [del_d for i in range(30)] + [math.cos(del_heading) for j in range(15)] + [math.sin(del_heading) for k in range(15)]

    def callback(self, data):
        self.observe = list(data.ranges)

    def callback2(self, data):
        if self.name == 'test_dagger_robot_green':
            pose = data.pose[1]
        elif self.name == 'test_dagger_robot_red':
            pose = data.pose[2]

        self.x = pose.position.x
        self.y = pose.position.y
        self.heading = math.atan2(2 * (pose.orientation.w * pose.orientation.z + pose.orientation.x *
                                       pose.orientation.y),(1 - 2*(pose.orientation.y ** 2 + pose.orientation.z ** 2)))

    def update_target(self, pose):
        self.target = pose

    def getModelState(self, command):
        myModelState = ModelState()
        if self.name == 'test_dagger_robot_red':
            myModelState.model_name = 'test_dagger_robot_red'
            myModelState.reference_frame = 'test_dagger_robot_red'
        elif self.name == 'test_dagger_robot_green':
            myModelState.model_name = 'test_dagger_robot_green'
            myModelState.reference_frame = 'test_dagger_robot_green'

        if command == 0:
            if self.v < 1.5:
                self.v += 0.1
            else:
                self.v = self.v

        elif command == 1:
            if self.v > 0:
                self.v -= 0.1
            else:
                self.v = 0
        elif command == 2:
            self.yaw += 0.05
        elif command == 3:
            self.yaw -= 0.05
        else:
            pass

        if self.yaw > math.pi:
            self.yaw -= 2 * math.pi
        elif self.yaw < -math.pi:
            self.yaw += 2 * math.pi

        myModelState.twist.linear.x = self.v

        myQuaternion = Quaternion()
        myQuaternion.w = math.cos(self.yaw / 2)
        myQuaternion.z = math.sin(self.yaw / 2)
        myModelState.pose.orientation = myQuaternion
        self.yaw = 0

        return myModelState

    def tick(self, command):
        modelstate = self.getModelState(command)
        self.pub.publish(modelstate)