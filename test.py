import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

from envs.rendezvous import Poliastro_env


env=Poliastro_env()

env=DummyVecEnv([lambda: env])
model=PPO("MlpPolicy", env, verbose=1)