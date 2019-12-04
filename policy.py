import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random


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


class Policy(object):
    def __init__(self):
        self.memory_pointer = 0
        self.memory_size = 3000
        self.batch_size = 16
        self.state_buffer = []
        self.label_buffer = []
        self.LR = 0.0001
        self.net = FCNet(180, 60, 256, 64, 256, 4)
        self.train_num = 0

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.LR)
        self.loss_func = nn.CrossEntropyLoss()

    def choose_action(self, state):
        action = self.net(torch.tensor([state[0:180]]).float(), torch.tensor([state[180:]]).float())

        return action

    def store_transition(self, state, label):
        if self.memory_pointer < self.memory_size:
            self.state_buffer.append(state)
            self.label_buffer.append(label)
        else:
            self.state_buffer[self.memory_pointer % self.memory_size] = state
            self.label_buffer[self.memory_pointer % self.memory_size] = label

        self.memory_pointer += 1

    def sample_batch(self):
        index_list = [random.randint(0, self.memory_size-1) for i in range(self.batch_size)]
        sample_state = []
        sample_label = []
        for index in index_list:
            sample_state.append(self.state_buffer[index])
            sample_label.append(self.label_buffer[index])

        return sample_state, sample_label

    def learn(self):
        if self.memory_pointer < self.memory_size:
            pass
        else:
            sample_state, sample_label = self.sample_batch()
            sample_state = torch.tensor(sample_state).float()
            output = self.net(sample_state[:, 0:180], sample_state[:, 180:])
            sample_label = torch.tensor(sample_label)
            loss = self.loss_func(output + 1e-8, sample_label)  # cross entropy loss
            if self.train_num % 100 == 0:
                print(sample_state)
                print("loss:", loss)
                print("output:", output)

            self.optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            self.optimizer.step()
            self.train_num += 1

    def save(self):
        torch.save(self.net.state_dict(), 'net_parameters.pkl')