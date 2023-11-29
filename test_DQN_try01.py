'''
'''
import sys
sys.path.append('.')
sys.path.append('.DQN_try01')

import numpy as np
import pandas as pd
import random
import torch
import gymnasium as gym
import os
import torch.utils.tensorboard as tb

from DQN_try01 import Utils
from DQN_try01.Agents import *
from DQN_try01.Trainer import Trainer
from DQN_try01.Writer import *
from DQN_try01.Args import Args
from DQN_try01.Nets import *
from EnvTool import *

if __name__ == "__main__":
    # Hyper Parameters
    args = Args(comment="DQN")
    # set writer
    writer = Writer(args)
    # set random seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Env: CartPole-v0
    node_num = 5
    file_name = "solomon_100\\R101.txt"
    # env = EnvESPPRC(file_name)
    # env = EnvVRPTW(file_name, lmt_node_num=node_num)
    env = EnvTSP(file_name, lmt_node_num=node_num)
    # train
    agent = DQNAgent(node_num+1, node_num, args, writer)
    trainer = Trainer(env, agent, args, writer)
    trainer.train()
