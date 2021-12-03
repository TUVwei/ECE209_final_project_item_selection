from utils import feature_extract
from kgEnv import kgEnv
from DQN import DQN
import numpy as np
import torch


def get_state(env):
    # get a state
    graph_state = env.get_adj()
    # extract the features for state
    x, _, _ = feature_extract(graph_state)
    edge_index = np.vstack(np.where(graph_state[1]))
    edge_index = torch.tensor(edge_index, dtype=torch.int64, requires_grad=False)
    x = torch.tensor(x, dtype=torch.float32, requires_grad=False)
    state = [x, edge_index]
    return state


total_steps = 0
N_NODES = 100
env = kgEnv(N_NODES)
# env.plot_graph(0)
dqn = DQN()
for ite in range(1, 2001):
    done_flag = False
    state = get_state(env)
    while not done_flag:
        # predict
        Q = dqn.predict(state=state)
        action, selectedQ = dqn.select(Q)
        # move
        n, r, _ = env.remove_node(action)
        # env.plot_graph(1)
        total_steps += 1
        # next sate
        next_state = get_state(env)
        # train
        dqn.train(state, action, r, next_state)
        # next
        state = next_state
        # if done
        if r == 10:
            done_flag = True
    # print
    dqn.update()
    if ite%50 == 0:
        print("Here is the {} iteration, the average steps are: {:.2f}".format(ite, total_steps/50))
        total_steps = 0
    env.reset_all()

