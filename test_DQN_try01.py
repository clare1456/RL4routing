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
from DQN_try01.StateBuilder import *
from EnvTool import *

if __name__ == "__main__":
    # Hyper Parameters
    args = Args(comment="DQN")
    args.run_code = args.date + "_feasible_try_4_Routing_with_5P"
    args.log_path = os.path.join(args.root_path, "log", args.comment, args.run_code)
    args.model_save_path = os.path.join(args.root_path, "model", args.comment, args.run_code)
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
    # state setting
    # state_builder = StateBuilder_node01(node_num)
    # state_builder = StateBuilder_node01_mask(node_num)
    # state_builder = StateBuilder_node_now_xy(node_num)
    # state_builder = StateBuilder_node_all_xy(node_num)
    state_builder = StateBuilder_node01_mask_all_xy(node_num)
    # train
    state_space_dim = state_builder.get_state_dim()
    agent = DQNAgent(state_space_dim, node_num, args, writer)
    trainer = Trainer(env, agent, args, writer, state_builder)
    trainer.train()
