'''
File: \run.py
Project: Tianshou_test
Description: basic structure using tianshou
-----
Author: CharlesLee
Created Date: Thursday February 16th 2023
'''
#%%
from typing import Any, Dict, Optional, Sequence, Tuple, Union
from tianshou.data import Batch, ReplayBuffer, to_torch_as
import torch, numpy as np
import gym as gym
import tianshou as ts
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy, PGPolicy, DQNPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.discrete import Actor, Critic
from EnvTool import *
from graph_encoder import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%%
# 1. Environment
# create 2 vectorized environments both for training and testing
nodeNum = 10
env = EnvTSP(lmt_node_num=nodeNum)
train_envs = DummyVectorEnv([lambda: EnvTSP(lmt_node_num=nodeNum) for _ in range(20)])
test_envs = DummyVectorEnv([lambda: EnvTSP(lmt_node_num=nodeNum) for _ in range(10)])

#%%
# 2. Model
# net is shared head of the actor and the critic
# rewrite the forward function to realize mask 
class MyNet(torch.nn.Module):
    def __init__(
            self, 
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512,
        ):
        super().__init__()
        self.encoder = GraphAttentionEncoder(n_heads, embed_dim, n_layers, node_dim, normalization, feed_forward_hidden) 
        self.decoder = MultiHeadAttention(n_heads, embed_dim, embed_dim)
        self.output_dim = embed_dim * 2
    
    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        """Mapping: obs -> flatten (inside MLP)-> logits."""
        if isinstance(obs, torch.Tensor) == False:
            obs = torch.FloatTensor(obs)
        embedings, mean_embeding = self.encoder(obs)
        logits = self.decoder(embedings[:, :2], embedings[:, 2:])
        return logits, state

class MyActor(Actor):
    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        mask = torch.BoolTensor(info["mask"])
        if self.device is not None:
            obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        # logits, hidden = self.preprocess(obs.reshape(len(obs), -1), state)
        logits, hidden = self.preprocess(obs, state)
        logits = self.last(logits)
        logits[mask] = torch.tensor(-np.inf)
        if self.softmax_output:
            logits = torch.nn.functional.softmax(logits, dim=-1)
        return logits, hidden

class MyCritic(Critic):
    def forward(self, obs: Union[np.ndarray, torch.Tensor], **kwargs: Any) -> torch.Tensor:
        """Mapping: s -> V(s)."""
        if self.device is not None:
            obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        # logits, _ = self.preprocess(obs.reshape(len(obs), -1), state=kwargs.get("state", None))
        logits, _ = self.preprocess(obs, state=kwargs.get("state", None))
        return self.last(logits)

net = MyNet(n_heads=8, embed_dim=128, n_layers=1, node_dim=env.state_dim, feed_forward_hidden=256)
actor = MyActor(net, env.action_space.n, device=device).to(device)
critic = Critic(net, device=device).to(device)
actor_critic = ActorCritic(actor, critic)

# optimizer of the actor and the critic
optim = torch.optim.AdamW(actor_critic.parameters(), lr=1e-3)

#%%
# 3. Policy
class MyPolicy(PPOPolicy):
    def process_fn(self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray) -> Batch:
        batch = super().process_fn(batch, buffer, indices)
        # reward normalization
        batch.rew -= batch.rew.mean() 
        batch.rew /= batch.rew.std()
        return batch

distributions = torch.distributions.Categorical
policy = MyPolicy(actor, critic, optim, distributions, action_space=env.action_space, deterministic_eval=True)
# deterministic_eval=True means choose best action in evaluation

#%%
# 4. Collector
train_collector = Collector(policy, train_envs, VectorReplayBuffer(2000000, len(train_envs)), exploration_noise=True)
test_collector = Collector(policy, test_envs)

#%%
# 5. Trainer
result = onpolicy_trainer(
    policy, 
    train_collector, 
    test_collector, 
    max_epoch=100, 
    step_per_epoch=10000, 
    repeat_per_collect=10, 
    episode_per_test=10, 
    batch_size=256, 
    step_per_collect=500, 
    stop_fn=lambda mean_reward: mean_reward >= 800, 
)

# show result
print(result)

#%%
# watch performance
policy.eval()
result = test_collector.collect(n_episode=1, render=True)
print("final reward: {}, length: {}".format(result["rew"].mean(), result["lens"].mean()))












