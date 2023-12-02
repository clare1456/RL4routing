import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import tqdm
import time
from DQN_try01 import Utils

class Trainer:
    def __init__(self, env, agent, args, writer, state_builder):
        self.env = env
        self.agent = agent
        self.args = args
        self.writer = writer
        self.state_builder = state_builder

    def train(self):
        # set random seed
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        pbar = tqdm.tqdm(range(self.args.episode_num))
        d_epsilon = (self.args.epsilon_final - self.args.epsilon) / self.args.episode_num
        for episode_i in range(self.args.episode_num):
            if episode_i < self.args.episode_num - self.args.episode_4greedy:
                self.agent.epsilon += d_epsilon
            else:
                self.agent.epsilon = 0
            state_info, info = self.env.reset()
            state = self.state_builder.cal_state(state_info, info)
            episode_reward = 0
            env_interact_timecost = 0
            predict_timecost = 0
            update_timecost = 0
            route = [self.state_builder.cal_node(state, state_info, info)]
            while True:
                # sample action
                start = time.time()
                action = self.agent.take_action(state, info) 
                predict_timecost += time.time() - start
                # interact with env
                start = time.time()
                next_state_info, reward, done, info = self.env.step(action) 
                next_state = self.state_builder.cal_state(next_state_info, info)
                route.append(self.state_builder.cal_node(next_state, state_info, info))
                env_interact_timecost += time.time() - start
                episode_reward += reward
                # save experience and learn
                start = time.time()
                self.agent.save_experience(state, action, reward, next_state, done, info["mask"])
                update_timecost += time.time() - start 
                # log
                # Utils.log_info("common_log", "next_state_info = " + str(next_state_info))
                Utils.log_info("common_log", "next_state = " + str(next_state))
                Utils.log_info("common_log", "\n")
                # render
                if self.args.render_flag:
                    self.env.render()
                # check done
                if done:
                    break
                state = next_state
            # record
            episode_info = "Episode {}: Reward={}, route={} ".format(episode_i, round(episode_reward, 2), route)
            Utils.log_info("common_log", episode_info)
            Utils.log_info("common_log", "\n==================\n")
            Utils.log_info("episode_info", episode_info)
            pbar.update(1)
            pbar.set_description("Episode {}: Reward={}".format(episode_i, episode_reward))
            self.writer.add_scalar("env/episode_reward", episode_reward, episode_i)
            self.writer.add_scalar("timecost/interact_timecost", env_interact_timecost, episode_i)
            self.writer.add_scalar("timecost/predict_timecost", predict_timecost, episode_i)
            self.writer.add_scalar("timecost/learn_timecost", update_timecost, episode_i)
            self.writer.buffer_update()
            # save model
            if episode_i % self.args.model_save_episode == 0:
                self.agent.save_model()
        pbar.close()

