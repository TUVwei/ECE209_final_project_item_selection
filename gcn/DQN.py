import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as gnn


LR = 0.001
GAMMA = 0.9
N_NODES = 100


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = gnn.ChebConv(N_NODES, N_NODES*3, 2)
        self.conv2 = gnn.ChebConv(N_NODES*3, N_NODES*5, 2)
        self.fc1 = nn.Linear(N_NODES * 5, N_NODES * 2)
        self.fc2 = nn.Linear(N_NODES * 2, 1)

    def forward(self, data):
        x, edge_index = data
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class DQN():
    def __init__(self):  # define DQN
        self.eval_net, self.target_net = GCN(), GCN()
        self.EPSILON = 0.5
        self.decay_e = 0.003
        self.learn_step_counter = 0
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)  # optimization

    def predict(self, state):
        with torch.no_grad():
            Q = self.eval_net.forward(state)
        return Q.detach().numpy()

    def select(self, Q):
        if np.random.uniform() < self.EPSILON:
            action = np.random.choice(list(np.where(Q == max(Q))[0]), 1)[0]
        else:
            action = np.random.randint(0, N_NODES)
        selectedQ = Q[action]
        return action, selectedQ

    def train(self, state, action, utility, next_state):
        # target parameter update
        if self.learn_step_counter % 50 == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        q_eval = self.eval_net(state)
        q = q_eval[action]

        # next
        if utility >= 10:
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


