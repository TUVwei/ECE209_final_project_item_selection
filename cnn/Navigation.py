from utils import feature_extract
from kgEnv import kgEnv
from DQN import Network
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from DQN import DQN
from kgEnv import kgEnv
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


N_NODES = 100
dqnconHandler = DQN()
env = kgEnv(N_NODES)
env.plot_graph(0)
total_steps = 0
def get_state(env):
    # get a state
    graph_state = env.get_adj()
    # extract the features for state
    diag_mat,Lap_mat,dis_mat = feature_extract(graph_state)
    ft_mat = np.zeros((3,N_NODES, N_NODES))
    ft_mat[0, :, :] = diag_mat  # diagnol matrix
    ft_mat[1, :, :] = Lap_mat  # Laguassian matrix
    ft_mat[2, :, :] = dis_mat  # nodes matrix
    state = ft_mat
    state = torch.tensor(state, dtype=torch.float32, requires_grad=False)

    return state

for turn in range(1,5001):


    # generate environment
    done_flag = False
    state = get_state(env)
    while not done_flag:
        #predict
        allQ = dqnconHandler.predict(state)
        #print(allQ)
        selection, selectedQ = dqnconHandler.select(allQ)
       #print(selection)
        #print(selectedQ)

        _,utility,_ = env.remove_node(selection)


        # next state
        nextState = get_state(env)
        #print(n_graph[0])
        #print(selection)
        #print(state.shape)

        # train
        dqnconHandler.train(state, selection, utility, nextState)

        state = nextState

        if utility == 10:
            done_flag = True
        total_steps+=1


    dqnconHandler.update()
    if turn % 10 == 0:
        print("Here is the {} iteration, the average steps are: {:.2f}".format(turn, total_steps / 10))
        total_steps = 0
    env.reset_all()











