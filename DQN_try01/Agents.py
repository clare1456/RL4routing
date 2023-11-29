import numpy as np
import torch
import torch.nn.functional as F
import time
import os
from .Nets import *
from .ReplayBuffer import ReplayBuffer

class Agent:
    def __init__(self, state_space, action_space, args, writer):
        # read args
        self.state_space = state_space
        self.action_space = action_space
        self.args = args
        self.writer = writer
    
    def take_action(self, state):
        """ 
        epsilon-greedy policy for exploration
        """
        if np.random.uniform() < self.epsilon:
            action = self.random_action(state)
        else:
            action = self.greedy_action(state)
        return action
    
    def random_action(self, state):
        """
        explore action randomly for training
        """
        raise NotImplementedError
    
    def greedy_action(self, state):
        """ 
        predict action (greedy) for evaluation
        """
        raise NotImplementedError
    
    def save_experience(self, state, action, reward, next_state, done):
        """
        save experience in !!each step!!, and update
        """
        raise NotImplementedError
    
    def save_model(self):
        """ 
        save model into file 
        """
        raise NotImplementedError


class DQNAgent(Agent):
    def __init__(self, state_space, action_space, args, writer):
        super(DQNAgent, self).__init__(state_space, action_space, args, writer)
        # set properties
        self.epsilon = args.epsilon
        self.learning_rate = args.learning_rate
        self.q_net = QNet(self.state_space, self.args.hidden_size, self.action_space, self.args.device).to(self.args.device)
        self.target_q_net = QNet(self.state_space, self.args.hidden_size, self.action_space, self.args.device).to(self.args.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.q_net.train()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayBuffer(self.args.pool_size)
        self.step_cnt = 0
    
    def random_action(self, state, info):
        """ 
        explore action randomly for training
        """
        action = np.random.randint(0, self.action_space)
        return action
    
    def greedy_action(self, state, info):
        """ 
        predict action (greedy) for evaluation
        """
        q_values = self.q_net(state)
        action = torch.argmax(q_values).item()
        return action
    
    def save_experience(self, state, action, reward, next_state, done):
        """
        save experience in !!each step!!, and update
        """
        # save experience
        self.replay_buffer.add(state, action, reward, next_state, done)
        # update q network
        if self.step_cnt % self.args.update_steps == 0:
            self._update()
            self._update_coeff()
        # syncronize target network
        if self.step_cnt % self.args.sync_target_steps == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        # update step count
        self.step_cnt += 1

    def _update(self):
        # train q network with experience replay
        if len(self.replay_buffer.memory) >= self.args.warmup_size:
            # sample experiences
            start = time.time()
            s_list, a_list, r_list, _s_list, done_list = self.replay_buffer.sample(self.args.batch_size)
            s_tensor = torch.tensor(s_list, dtype=torch.float32, device=self.args.device)
            a_tensor = torch.tensor(a_list, dtype=torch.long, device=self.args.device)
            r_tensor = torch.tensor(r_list, dtype=torch.float32, device=self.args.device)
            _s_tensor = torch.tensor(_s_list, dtype=torch.float32, device=self.args.device)
            done_tensor = torch.tensor(done_list, dtype=torch.float32, device=self.args.device)
            sample_timecost = time.time() - start
            # calculate loss
            start = time.time()
            predict_value = self.q_net(s_tensor).gather(1, a_tensor.unsqueeze(1)).squeeze(1)
            target_value = self._get_target_value(s_tensor, a_tensor, r_tensor, _s_tensor, done_tensor)
            loss = torch.mean(F.mse_loss(predict_value, target_value.detach()))
            cal_loss_timecost = time.time() - start
            # update q network
            start = time.time()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            backward_loss_timecost = time.time() - start
            # record information
            if self.args.log_flag:
                # record timecost
                self.writer.add_scalar_to_buffer("timecost/update-sample_timecost", sample_timecost, self.step_cnt)
                self.writer.add_scalar_to_buffer("timecost/update-cal_loss_timecost", cal_loss_timecost, self.step_cnt)
                self.writer.add_scalar_to_buffer("timecost/update-backward_loss_timecost", backward_loss_timecost, self.step_cnt)
                # record loss
                self.writer.add_scalar_to_buffer("net/loss", loss.detach().cpu(), self.step_cnt)
                # record q value
                self.writer.add_scalar_to_buffer("net/mean_q_value", torch.mean(predict_value).detach().cpu(), self.step_cnt)
                # record grad
                self.writer.add_scalar_to_buffer("net/mean_grad", torch.mean(torch.abs(self.q_net.fc1.weight.grad)).cpu(), self.step_cnt) 
                self.writer.add_scalar_to_buffer("net/max_grad", torch.max(torch.abs(self.q_net.fc1.weight.grad)).cpu(), self.step_cnt)
                self.writer.add_scalar_to_buffer("net/min_grad", torch.min(torch.abs(self.q_net.fc1.weight.grad)).cpu(), self.step_cnt)
                self.writer.add_scalar_to_buffer("net/std_grad", torch.std(torch.abs(self.q_net.fc1.weight.grad)).cpu(), self.step_cnt)
     
    @torch.no_grad()
    def _get_target_value(self, s_tensor, a_tensor, r_tensor, _s_tensor, done_tensor):
        # calculate target value
        return r_tensor + self.args.gamma * torch.max(self.target_q_net(_s_tensor), dim=1)[0] * (1-done_tensor)
    
    def _update_coeff(self):
        # update coefficents
        if self.step_cnt % 100 == 0:
            self.epsilon = self.epsilon * 0.99
            self.learning_rate = self.learning_rate * 0.998
        # record coefficents
        self.writer.add_scalar_to_buffer("coeff/epsilon", self.epsilon, self.step_cnt)
        self.writer.add_scalar_to_buffer("coeff/learning_rate", self.learning_rate, self.step_cnt)
     
    def save_model(self):
        """ 
        save model into file 
        """
        if self.args.save_model_flag:
            if not os.path.exists(self.args.model_save_path):
                os.makedirs(self.args.model_save_path)
            self.q_net.save(self.args.model_save_path)
         

