'''
File: \\EnvTool.py
Project: ML4ESPPRC
Description: Environment for trainning and relevant tools
-----
Author: CharlesLee
Created Date: Monday February 20th 2023
'''

import torch, numpy as np
import matplotlib.pyplot as plt
import random
import gym as gym
from GraphTool import Graph

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class EnvESPPRC(gym.Env):
    def __init__(self, file_name="solomon_100\\R101.txt", lmt_node_num=None):
        super().__init__()
        self.graph = Graph(file_name, limit_node_num=lmt_node_num) 
        self.state_dim = 7 # state dimention: x, y, e, l, s, dual
        self.action_space = gym.spaces.Discrete(self.graph.nodeNum)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, ((self.graph.nodeNum+2), self.state_dim), dtype=np.float32)
        self.dualValue = np.arange(self.graph.nodeNum)
        self.end_pos = 0 # terminal position, depot index, stay static
        self.cur_pos = 0 # initial position, varies in process
        self.res_capacity = self.graph.capacity # residual capacity
        self.cur_time = 0 # current time
        self.routes = [[0]] # record routes
        self.check_demand = True
        self.check_timeWindow = True

        """ initialize state """
        # state: (cur_state, end_state, all_node_states) (103 x state_dim)
        self.cur_state = np.zeros((2+self.graph.nodeNum, self.state_dim), dtype=np.float32)
        for i in range(self.graph.nodeNum+2):
            if i == 0:
                idx = self.cur_pos
            elif i == 1:
                idx = self.end_pos
            else:
                idx = i - 2
            self.cur_state[i] = self._get_node_state(idx)
    
    def _get_node_state(self, idx):
        """_get_node_state 

        Args:
            idx (int): idx of node

        Returns:
            node_state (ndarray): state of node
        """
        node_state = np.zeros(self.state_dim, dtype=np.float32) 
        node_state[:2] = self.graph.location[idx] # location
        node_state[3] = self.graph.readyTime[idx] # ready time
        node_state[4] = self.graph.dueTime[idx] # due time
        node_state[5] = self.graph.serviceTime[idx] # service time
        node_state[6] = self.dualValue[idx] # dual value
        return node_state    

    def reset(self):
        """reset reset each episode
        """
        self.visited = np.zeros(self.graph.nodeNum, dtype=bool)
        self.cur_pos = 0 
        self.res_capacity = self.graph.capacity # residual capacity
        self.cur_time = 0 # current time
        self.routes = [[0]] # record routes
        self._set_state(self.cur_pos)
        info = {}
        info["mask"] = self._get_mask()
        return self.cur_state, info

    def _set_state(self, cur_pos):
        """_set_state set state each episode, only need to set cur_pos state

        Args:
            cur_pos (int): index of current node
        """
        self.cur_state[0] = self._get_node_state(cur_pos)
 
    def step(self, action):
        reward = 0
        done = False
        info = {}
        # Check feasibility
        if self._check_infeasible(action):
            reward = -1000
            done = True
            info["mask"] = self._get_mask()
            return self.cur_state, reward, done, info
        # Update constraint variables
        if action == 0:
            self.res_capacity = self.graph.capacity
            self.cur_time = 0
        else:
            self.res_capacity -= self.graph.demand[action]
            self.cur_time = max(self.graph.readyTime[action], 
                                self.cur_time + self.graph.timeMatrix[self.cur_pos, action])
        # Reward = dualValue - distance
        reward = self.reward_function(action)
        # Set visited (ignore 0)
        if action != 0:
            self.visited[action] = 1
        # Set cur_pos, state
        self.cur_pos = action
        self._set_state(self.cur_pos)
        # End game when choose 0 
        if self.stop_condition(action):
            done = True
        # record routes
        self.routes[-1].append(action)
        if done == False and action == 0:
            self.routes.append([0])
        if done == True:
            self.routes[-1].append(0)
            reward += self.end_reward_function(action)
        # get mask
        info["mask"] = self._get_mask()
        return self.cur_state, reward, done, info

    def _check_infeasible(self, next_node):
        """_check_feasible ( 0 -> feasible, 1 -> infeasible
            infeasible conditions: 
            1. visited / repeat cur_pos
            2. break capacity constraint
            3. break time window
        )

        Args:
            next_node (int): index of node to check

        Returns:
            is_infeasible (bool): infeasible or not
        """
        return (next_node == self.cur_pos or self.visited[next_node] == 1 or 
                (self.check_demand and self.res_capacity - self.graph.demand[next_node] < 0) or
                (self.check_timeWindow and self.cur_time + self.graph.timeMatrix[self.cur_pos, next_node] > self.graph.dueTime[next_node]))

    def _get_mask(self, check_feasibility=True):
        # mask visited nodes
        mask = self.visited.copy()
        mask[self.cur_pos] = 1 # avoid repeat choice
        # mask infeasible nodes
        if check_feasibility:
            for idx in range(self.graph.nodeNum):
                if mask[idx] == 1:
                    continue
                if self._check_infeasible(idx):
                    mask[idx] = 1
        if mask.sum() == len(mask):
            mask[0] = False
        return mask

    def stop_condition(self, action):
        return action == 0
    
    def reward_function(self, action):
        return self.dualValue[action] - self.graph.disMatrix[self.cur_pos, action]

    def end_reward_function(self, action):
        return -self.graph.disMatrix[action, 0]

    def render(self, mode="human", close=False):
        self.graph.render(self.routes) 

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def human_test(self, render=False):
        """human_test (test env with human iteraction)
        """
        print("Game Start!")
        is_stop = False
        while is_stop == False:
            done = 0
            state, info = self.reset()
            mask = info['mask']
            step_cnt = 0
            reward_sum = 0
            while done != 1:
                print("Step {}".format(step_cnt))
                choices = [idx for idx in range(self.graph.nodeNum) if mask[idx] == 0]
                print("  Current choices: {}".format(choices))
                action = input("  Please type in your action: ")
                while True:
                    try:
                        action = int(action)
                        break
                    except:
                        action = input("  Please type right action: ") 
                state, reward, done, info = self.step(int(action))
                mask = info["mask"].tolist()
                reward_sum += reward
                print("  reward = {:.2f}, done = {}".format(reward, done))
                print("  routes = {}".format(self.routes))
                step_cnt += 1
            print("Game Over, total reward = {}".format(reward_sum))
            if render:
                self.render()
            flag = input("\nTry again? (yes / no)")
            is_stop = flag == "no"

class EnvVRPTW(EnvESPPRC):
    def stop_condition(self, action):
        return sum(self.visited) == self.graph.nodeNum - 1

    def reward_function(self, action):
        return -self.graph.disMatrix[self.cur_pos, action]

class EnvTSP(EnvVRPTW):
    def reset(self):
        self.check_demand = False
        self.check_timeWindow = False
        state, info = super().reset()
        self.visited[0] = True
        info['mask'] = self._get_mask()
        return state, info
    
    def stop_condition(self, action):
        return sum(self.visited) == self.graph.nodeNum

if __name__ == "__main__":
    file_name = "solomon_100\\R101.txt"
    # env = EnvESPPRC(file_name)
    # env = EnvVRPTW(file_name, lmt_node_num=5)
    env = EnvTSP(file_name, lmt_node_num=5)
    # human interact test
    env.human_test(render=False)
    




