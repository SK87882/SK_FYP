import os
import airsim
import numpy as np
import math
import torch
import time
from argparse import ArgumentParser
import gym
from airgym.envs.newtrain import AirSimEnv
from gym import spaces
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, EveryNTimesteps, CheckpointCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.vec_env import DummyVecEnv, make_vec_env
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor

def callback_function(locals_dict, globals_dict):
    
    # Access the current statistics
    current_episode_reward = locals_dict['episode_reward']
    current_step = locals_dict['total_timesteps']
    current_policy_loss = locals_dict['policy_loss']
    current_value_loss = locals_dict['value_loss']
    current_q_loss = locals_dict['q_loss']
    ep_rew_mean = locals_dict['ep_rew_mean']
    current_ep_rew_mean = np.mean(locals_dict['ep_rew_mean'])
    current_cumulative_reward = locals_dict['cumulative_reward']
    
    #logs the statistics to Tensorboard
    print(f'Timestep: {current_step}, Mean episode reward: {current_ep_rew_mean:.2f}, Mean current episode reward: {ep_rew_mean:.2f}, Cumulative reward: {current_cumulative_reward:.2f}, Policy loss: {current_policy_loss:.2f}, Value loss: {current_value_loss:.2f}, Q loss: {current_q_loss:.2f}')

# Define the directory to save the model and Tensorboard logs
mon_dir = "./tb_logs/"
log_dir = "./tb_logs/TD3"

#check the log directory exists
os.makedirs(log_dir, exist_ok=True)

# Create a list of target positions
target_pos = airsim.Vector3r(5.5,0,-3)


#Create an instance of custom Airsim Environment
env = AirSimEnv(target_pos)

#Wrapping env in Monitor to allow for logging
env = Monitor(env, mon_dir)

#Dummy vectorized env. Required for compatibility with Stable Baselines 3
env = DummyVecEnv([lambda: env])

#Define TD3 Hyperparameters
td3_kwargs = {
    "policy_delay": 1,
    "gamma": 0.99,
    "learning_rate": 0.0003,
    "batch_size": 256,
    "buffer_size": 10000,
    "learning_starts": 10000,
    "gradient_steps": 1000,
    "train_freq": 1000,
    "tau": 0.005
}

#creates an instance of the TD3 algorithm with MLP
model = TD3(policy="MlpPolicy", env=env, **td3_kwargs, tensorboard_log=log_dir)

#This creates a callback that saves the model every epoch 
checkpoint_callback = CheckpointCallback(save_freq=1, save_path=log_dir)

#Evaluation callback that evaluates the model every epoch
eval_callback = EvalCallback(env, callback_on_new_best=None, eval_freq=1, best_model_save_path=log_dir,
                             log_path=log_dir, n_eval_episodes=2, deterministic=True, render=False,
                             verbose=0)

#Creating method of the RLA object
model.learn(total_timesteps=100000, callback=[checkpoint_callback, eval_callback])

#Save the model
model.save("TD3_Drone_Training")

#Load the model
model = TD3.load("TD3_Drone_Training")
