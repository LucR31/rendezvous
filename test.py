import gym
import importlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#from baselines
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
#from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C

#local
from envs.rendezvous import Poliastro_env
from envs.env_test import play_game, play_game_train


env=Poliastro_env(r_1= [-6045, -5510, 2504],v_1= [-3.357, 5.728, 2.133])

model=PPO("MultiInputPolicy", env, verbose=0)
model.learn(total_timesteps=40000)