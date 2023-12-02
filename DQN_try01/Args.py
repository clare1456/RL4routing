import os
import pandas as pd
import numpy as np
import sys
sys.path.append('.')
from . import Utils

class Args:
    def __init__(self, comment=None):
        """ Flags """
        self.render_flag = False         # show UI or not
        self.log_flag = True             # log or not
        self.save_model_flag = True      # save model or not
        """ Train """
        self.env_name = "CartPole-v1"    # env name
        self.episode_num = 1000           # number of episodes
        self.episode_4greedy = 100        # number of episodes for pure greedy
        self.pool_size = 100000           # capacity of experience replay
        self.warmup_size = 1000          # number of warmup steps
        self.batch_size = 128             # batch size
        self.hidden_size = 128         # number of neurons in hidden layer
        self.update_steps = 20           # steps to update q network
        self.learning_rate = 1e-3       # learning rate
        self.sync_target_steps = 20      # steps to sync target network
        self.epsilon = 0.6               # greedy policy's first epsilon
        self.epsilon_final = 0.1         # greedy policy's final epsilon
        self.gamma = 0.7                 # reward discount
        self.seed = 1                    # random seed
        self.device = "cpu"              # device
        """ File """
        self.comment = "test" if comment is None else comment
        self.root_path = Utils.get_root_path()
        self.timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
        self.date = pd.Timestamp.now().strftime("%Y%m%d")
        self.run_code = self.timestamp
        self.log_path = os.path.join(self.root_path, "log", self.comment, self.run_code)
        self.model_save_path = os.path.join(self.root_path, "model", self.comment, self.run_code)
        self.model_save_episode = 100  # save model every n episodes