import airsim
import numpy as np
import gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback


class AirSimEnv(gym.Env):
    def __init__(self, target_pos):
        self.target_pos = target_pos
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.action_space = gym.spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]), dtype='float64')
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype='float64')
        self.max_distance = 150
        self.max_steps = 10000
        self.step_count = 0
        self.collided = False

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        self.step_count = 0
        quad_state = self.client.getMultirotorState().kinematics_estimated
        self.episode_reward = 0
        obs = np.array([quad_state.position.x_val, quad_state.position.y_val, quad_state.position.z_val,
                        quad_state.linear_velocity.x_val, quad_state.linear_velocity.y_val, quad_state.linear_velocity.z_val,
                        self.get_distance()])
        self.collided = False
        return obs

    def step(self, action):
        self.step_count += 1
        self.collision_threshold = 1
        pitch, roll, yaw = action
        quad_offset = airsim.to_quaternion(pitch, roll, yaw)
        quad_state = self.client.getMultirotorState().kinematics_estimated
        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        quad_vel += airsim.Vector3r(0, 0, 1)
       
        self.client.moveByRollPitchYawZAsync(
            roll=roll, pitch=pitch, yaw=yaw, 
            z=1, 
            duration=1, 
            #drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom
            ).join()

        self.client.moveByVelocityAsync(quad_vel.x_val, quad_vel.y_val, quad_vel.z_val, 1).join()
        obs = np.array([quad_state.position.x_val, quad_state.position.y_val, quad_state.position.z_val,
                    quad_state.linear_velocity.x_val, quad_state.linear_velocity.y_val, quad_state.linear_velocity.z_val,
                    self.get_distance()])
        reward = self.calculate_reward(obs)
        done = self.check_done(obs)
        info = {}

        collision_info = self.client.simGetCollisionInfo()  # get collision info
        if collision_info.has_collided:  # check if collision has occurred
            self.collided = True  # set collision flag
            reward -= 50  # give a negative reward for colliding       

        done = self.check_done(obs)

        
        return obs, reward, done, info

    def check_done(self, obs):
        if self.collided:  # terminate episode if collided
            return True
        if obs[6] < 0.5 or self.step_count > self.max_steps:
            return True
        target_pos = self.target_pos
        if obs[0] > 50 or obs[1] > 50 or obs[2] > 50:
            return True
        else:
            return False

    def calculate_reward(self, obs):
        #calculating the intial distance between the quadrotor and the target position
        initial_distance = self.get_distance()
        #calculates the current distance between from the quadrotor and the target
        current_distance = obs[6]
        #calculating the difference between the two
        distance_difference = initial_distance - current_distance
        
        if current_distance > 25:
            reward = -15
        else:
            reward = np.tanh(-distance_difference / 10.0)

        return float(reward)

    def get_distance(self):
        quad_state = self.client.getMultirotorState().kinematics_estimated
        quad_pos = quad_state.position
        target_pos = self.target_pos
        distance = np.linalg.norm(np.array([quad_pos.x_val, quad_pos.y_val, quad_pos.z_val]) - np.array([target_pos.x_val, target_pos.y_val, target_pos.z_val]))

        return distance

    


   



