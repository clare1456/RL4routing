import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import tqdm
import time
from DQN_try01 import Utils

class Trainer:
    def __init__(self, env, agent, args, writer):
        self.env = env
        self.agent = agent
        self.args = args
        self.writer = writer

    def train(self):
        # set random seed
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        pbar = tqdm.tqdm(range(self.args.episode_num))
        for episode_i in range(self.args.episode_num):
            state_info, info = self.env.reset()
            state = Utils.cal_state(state_info)
            episode_reward = 0
            episode_step = 0
            env_interact_timecost = 0
            predict_timecost = 0
            update_timecost = 0
            route = [Utils.cal_node(state)]
            while True:
                # sample action
                start = time.time()
                action = self.agent.take_action(state) 
                predict_timecost += time.time() - start
                # interact with env
                start = time.time()
                next_state_info, reward, done, info = self.env.step(action) 
                next_state = Utils.cal_state(next_state_info)
                route.append(Utils.cal_node(next_state))
                env_interact_timecost += time.time() - start
                episode_reward += reward
                episode_step += 1
                # save experience and learn
                start = time.time()
                self.agent.save_experience(state, action, reward, next_state, done)
                update_timecost += time.time() - start 
                # render
                if self.args.render_flag:
                    self.env.render()
                # check done
                if done:
                    break
                state = next_state
            # record
            episode_info = "Episode {}: Reward={}, route={} ".format(episode_i, round(episode_reward, 2), route)
            Utils.log_info("episode_info", episode_info)
            pbar.update(1)
            pbar.set_description("Episode {}: Reward={}".format(episode_i, episode_reward))
            self.writer.add_scalar("env/episode_reward", episode_reward, episode_i)
            self.writer.add_scalar("env/episode_step", episode_step, episode_i)
            self.writer.add_scalar("timecost/interact_timecost", env_interact_timecost, episode_i)
            self.writer.add_scalar("timecost/predict_timecost", predict_timecost, episode_i)
            self.writer.add_scalar("timecost/learn_timecost", update_timecost, episode_i)
            self.writer.buffer_update()
            # save model
            if episode_i % self.args.model_save_episode == 0:
                self.agent.save_model()
        pbar.close()

