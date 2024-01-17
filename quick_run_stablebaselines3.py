# File: quick_run_stablebaselines3.py
# Project: RL4routing
# File Created: 2024/1/16 20:04
# Author: limingzhe (lmz22@mails.tsinghua.edu.cn)

#%% check env
# from EnvTool import EnvESPPRC
# from stable_baselines3.common.env_checker import check_env
#
# env = EnvESPPRC(lmt_node_num=10)
# check_env(env)


#%% train
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from EnvTool import EnvESPPRC, EnvTSP, EnvVRPTW

# env = EnvESPPRC(lmt_node_num=10)
env = EnvTSP(lmt_node_num=10)
model = PPO("MlpPolicy", env, verbose=1)
log_path = "log/sb3/ppo_tsp"
logger = configure(log_path, ["tensorboard", "stdout"])
model.set_logger(logger)
model.learn(total_timesteps=100000)



