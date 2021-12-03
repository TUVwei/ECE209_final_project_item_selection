import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

N_NODES = 100
LR = 0.001
GAMMA = 0.9


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        #self.nn1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), padding=1, stride=(1,1), bias=True)
        #self.nn1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(10,10), padding=4, stride=(2,2), bias=True)
        self.nn1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, stride=(1, 1), bias=True)
        self.maxpool1 = nn.MaxPool2d(2,2)
        self.nn2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, stride=(1, 1), bias=True)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        #self.nn3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, stride=(1, 1), bias=True)
        #self.maxpool3 = nn.MaxPool2d(2, 2)
        #self.fc1 = nn.Linear(int(((N_NODES**2)/4)*32), 128)
        self.fc1 = nn.Linear(40000, 2560)
        self.fc1.weight.data.normal_(0, 0.1)
        #self.out = nn.Linear(128, N_NODES)
        self.out = nn.Linear(2560, N_NODES)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.nn1(x))
        x = self.maxpool1(x)
        x = F.relu(self.nn2(x))
        x = self.maxpool2(x)
       # x = F.relu(self.nn3(x))
        #x = self.maxpool3(x)
        x = torch.flatten(x)
        x = F.relu(self.fc1(x))
        actions_value = self.out(x)
        return actions_value


class DQN():
    def __init__(self):  # define DQN
        self.eval_net, self.target_net = Network(), Network()
        self.EPSILON = 0.5
        #self.decay_e = 0.0004
        self.decay_e = 0.004
        self.learn_step_counter = 0
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)  # optimization

    def predict(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        Q = self.eval_net.forward(state)
        Q = Q.tolist()
        #Q = Q[0]
        return Q

    def select(self, Q):
        if np.random.uniform() < self.EPSILON:
            action = np.random.choice(list(np.where(np.array(Q) == max(Q))[0]), 1)[0]
        else:
            action = np.random.randint(0, N_NODES)
        selectedQ = Q[action]
        return action, selectedQ

    def train(self, state, action, utility, next_state):
        # target parameter update
        if self.learn_step_counter % 50 == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        next_state = torch.unsqueeze(torch.FloatTensor(next_state), 0)

        q_eval = self.eval_net(state)

        q = q_eval

        q = q[action]

        # next
        if utility == 10:
            nextQ = 10
        else:
            nextQ = self.target_net(next_state).detach()
            nextQ = torch.max(nextQ)

        q_target = utility + GAMMA * nextQ
        loss = (q_target - q)**2
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update(self):
        self.EPSILON += self.decay_e


