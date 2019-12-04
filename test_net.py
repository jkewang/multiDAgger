import torch
import torch.nn.functional as F
import torch.nn as nn
import math

class FCNet(nn.Module):
    def __init__(self, inputs, inputs2, fc1, fc1_1, fc2, out):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(inputs, fc1)
        self.fc1_2 = nn.Linear(inputs2, fc1_1)
        self.fc2 = nn.Linear(fc1+fc1_1, fc2)
        #self.fc3 = nn.Linear(fc2, fc3)
        self.out = nn.Linear(fc1, out)

    def forward(self, x, x2):
        x = F.relu(self.fc1(x))
        x2 = F.relu(self.fc1_2(x2))
        x = torch.cat([x, x2], 1)
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        x = F.softmax(self.out(x))
        return x

class Testnet(object):
    def __init__(self):
        self.net = FCNet(180, 60, 256, 64, 256, 4)
        self.net.load_state_dict(torch.load("net_parameters.pkl"))

    def choose_action(self, state):
        action = self.net(torch.tensor([state[0:180]]).float(), torch.tensor([state[180:]]).float())

        return action