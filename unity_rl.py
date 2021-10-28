import numpy as np

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import gym
from gym import Env
from gym.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


class MyEnvironment(Env):
    def __init__(self, UnityEnv, dest):
        self.UnityEnv = UnityEnv
        self.behavior_name = list(UnityEnv.behavior_specs)[0]
        self.action_space = Box(low=np.array([-1, 150, 150]), high=np.array([1, 150, 150]))
        self.observation_space = Box(low=np.array([0, 0, 0, 0, 0]),
                                     high=np.array([20, 20, 20, 20, 20]))
        decision_steps, _ = self.UnityEnv.get_steps(self.behavior_name)
        self.initial_state = decision_steps.obs[0][0, -5:]
        self.state = self.initial_state
        self.origin = decision_steps.obs[0][0, :3]
        self.dest = dest
        self.pos = decision_steps.obs[0][0, :3]
        self.prev_dist = 0
        self.total_reward = 0
        self.time_length = 0

    def step(self, action):
        self.UnityEnv.set_actions(behavior_name, action.reshape(1, 3))
        decision_steps, terminal_steps = self.UnityEnv.get_steps(behavior_name)
        self.state = decision_steps.obs[0][0, -5:]
        self.pos = decision_steps.obs[0][0, :3]
        distance_from_origin = (((self.pos-self.origin) ** 2).sum()) ** 0.5
        distance_from_dest = (((self.pos-self.dest) ** 2).sum()) ** 0.5
        at_origin = abs(self.prev_dist - distance_from_origin) >= 5
        at_dest = distance_from_dest <= 5
        if at_origin:
            reward = -500
            done = True
            print("Crashed")
        else:
            reward = 1
            done = False
        if at_dest:
            reward = 5000
            print("Finished")
            env.reset()
        self.prev_dist = distance_from_origin
        self.total_reward += reward
        # print(self.total_reward)
        self.UnityEnv.step()
        info = {}

        return self.state, reward, done, info

    def reset(self):
        env.reset()
        self.state = self.initial_state
        self.time_length = 0
        print(f"Total Reward : {self.total_reward}")
        self.total_reward = 0
        return self.state


# Set your file name here
# ==============================================================================================================
file_dir = '/Road1/Prototype 1'
road_num = int(file_dir[file_dir.index('Road') + 4]) # Get Road Number from str file_dir
# ==============================================================================================================

channel = EngineConfigurationChannel()
env = UnityEnvironment(file_name=file_dir, side_channels=[channel])
channel.set_configuration_parameters(time_scale=10000)
env.reset()
behavior_name = list(env.behavior_specs)[0]
decision_steps, terminal_steps = env.get_steps(behavior_name)
cur_obs = decision_steps.obs[0][0, :]
origin = cur_obs[0:3]

dest_list = np.array([[124.08, 0.16, 114.18],       # Position of destination in Road1
                      [118.58, 0.16, 9.61],         # Position of destination in Road2
                      [78.03, 0.16, 21.34]])        # Position of destination in Road3

test_env = MyEnvironment(env, dest_list[road_num-1]) # Initialize MyEnvironment (OpenAI Gym)

# Mode Selection
# =================================================================================================
training_mode = False
load_model = True
# =================================================================================================

if training_mode and load_model: # Continue train previous model
    model = PPO.load('PPO_v2_500000.zip', env=test_env, device='cpu')
    model.learn(total_timesteps=500000)
    model.save('PPO_v2_1M')
elif training_mode: # Create new model
    model = PPO("MlpPolicy", test_env, verbose=1, device='cpu', tensorboard_log="Logs")
    model.learn(total_timesteps=2000000)
    model.save('PPO_v3_2M')
else: # Evaluate model
    model = PPO.load('PPO_v3_2M.zip', env=test_env, device='cpu')
    channel.set_configuration_parameters(time_scale=5)
    evaluate_policy(model, test_env, n_eval_episodes=10)

env.close()
