import rospy
import random
from gazebo_msgs.msg import ModelState
from sensor_msgs.msg import LaserScan
import math
import torch

import agent
import policy
import human_controller


class Referee(object):
    def __init__(self):
        self.epsilon = 0.05
        self.green_safe = [30 for i in range(360)]
        self.red_safe = [30 for i in range(360)]

        rospy.init_node("referee", anonymous='True')
        self.kidnapper_pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
        rospy.Subscriber('/hokuyo_safe_green', LaserScan, self.callback)
        rospy.Subscriber('/hokuyo_safe_red', LaserScan, self.callback2)

        self.rate = rospy.Rate(20)
        self.time = 0

        self.agent_green = agent.Agent(name='green')
        self.agent_red = agent.Agent(name='red')
        self.expert = human_controller.Controller()

        self.target_green = "goal_box"
        self.target_red = "goal_box_0"
        self.green_target = [-5.8, 5.37]
        self.red_target = [5.53, -5.5]

        self.agent_green.update_target(self.green_target)
        self.agent_red.update_target(self.red_target)
        self.generate_target('green')
        self.generate_target('red')

        self.dagger_policy = policy.Policy()

    def reset_game(self):
        self.agent_green.reset()
        self.agent_red.reset()

        agent_red_pose_x, agent_red_pose_y = self.generate_random_pose()
        self.kidnapper(self.agent_red.name, [agent_red_pose_x, agent_red_pose_y])
        agent_green_pose_x, agent_green_pose_y = self.generate_random_pose()
        self.kidnapper(self.agent_green.name, [agent_green_pose_x, agent_green_pose_y])

        target_red_pose_x ,target_red_pose_y = self.generate_random_pose()
        target_green_pose_x, target_green_pose_y = self.generate_random_pose()
        self.kidnapper(self.target_green, [target_green_pose_x, target_green_pose_y])
        self.kidnapper(self.target_red, [target_red_pose_x, target_red_pose_y])

        self.green_safe = [30 for i in range(360)]
        self.red_safe = [30 for i in range(360)]

        self.agent_green.update_target(self.green_target)
        self.agent_red.update_target(self.red_target)
        self.generate_target('green')
        self.generate_target('red')

    def generate_random_pose(self):
        pose_x = (13.6 * random.random()) - 6.8
        pose_y = (13.6 * random.random()) - 6.8
        while abs(pose_x) + abs(pose_y) < 6:
            pose_x = (13.6 * random.random()) - 6.8
            pose_y = (13.6 * random.random()) - 6.8

        return pose_x, pose_y

    def generate_target(self, color):
        if color == 'green':
            pose = self.generate_random_pose()
            self.green_target = pose
            print('kidnapped:', self.target_green)
            self.kidnapper(self.target_green, pose)

        if color == 'red':
            pose = self.generate_random_pose()
            self.red_target = pose
            self.kidnapper(self.target_red, pose)

    def tick(self):
        self.agent_green.update_target(self.green_target)
        self.agent_red.update_target(self.red_target)
        safety = self.check_collision()
        if safety[0]:
            self.agent_green.reset()
            pose_x, pose_y = self.generate_random_pose()
            self.kidnapper(self.agent_green.name, [pose_x, pose_y])

        if safety[1]:
            self.agent_red.reset()
            pose_x, pose_y = self.generate_random_pose()
            self.kidnapper(self.agent_red.name, [pose_x, pose_y])

        self.check_goal()

        expert_action = self.expert.command
        green_state= self.agent_green.ask_state()
        self.dagger_policy.store_transition(green_state, expert_action)
        random_epsilon = random.random()
        if random_epsilon > self.epsilon:
            self.agent_green.tick(self.expert.command)
        else:
            self.agent_green.tick(torch.max(self.dagger_policy.choose_action(green_state)[0], 0)[1].item())

        red_state = self.agent_red.ask_state()
        self.agent_red.tick(torch.max(self.dagger_policy.choose_action(red_state)[0], 0)[1].item())
        self.dagger_policy.learn()
        if self.epsilon < 0.95:
            self.epsilon += 0.000015
        if self.time % 100 == 0:
            print(self.epsilon)

        self.expert.command = 0
        if self.dagger_policy.train_num % 3000 == 1:
            self.dagger_policy.save()

        self.rate.sleep()
        self.time += 1

    def check_collision(self):
        return [(min(self.green_safe) < 0.52) or (self.dist() < 1.05), (min(self.red_safe) < 0.52) or (self.dist() < 1.05)]

    def check_goal(self):
        if self.time % 100 == 0:
            print("goal_dist", math.sqrt((self.agent_green.x - self.green_target[0]) ** 2 + (self.agent_green.y - self.green_target[1]) ** 2))
        if math.sqrt((self.agent_green.x - self.green_target[0]) ** 2 + (self.agent_green.y - self.green_target[1]) ** 2) < 1.5:
            self.agent_green.update_target(self.green_target)
            self.generate_target('green')

        if math.sqrt((self.agent_red.x - self.red_target[0]) ** 2 + (self.agent_red.y - self.red_target[1]) ** 2) < 1.5:
            self.agent_red.update_target(self.red_target)
            self.generate_target('red')

    def kidnapper(self, target, pose):
        modelstate = ModelState()
        modelstate.model_name = target
        modelstate.pose.position.x = pose[0]
        modelstate.pose.position.y = pose[1]
        modelstate.pose.position.z = 0.125

        self.kidnapper_pub.publish(modelstate)
        self.rate.sleep()

    def dist(self):
        return math.sqrt((self.agent_green.x - self.agent_red.x) ** 2 + (self.agent_green.y - self.agent_red.y) ** 2)

    def callback(self, data):
        self.green_safe = data.ranges

    def callback2(self, data):
        self.red_safe = data.ranges

game = Referee()
while not rospy.is_shutdown():
    game.tick()