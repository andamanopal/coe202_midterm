import numpy as np
import os
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from gym import Env
from gym.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold


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
        distance_from_origin = (((self.pos - self.origin) ** 2).sum()) ** 0.5
        distance_from_dest = (((self.pos - self.dest) ** 2).sum()) ** 0.5
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
            self.UnityEnv.reset()
        self.prev_dist = distance_from_origin
        self.total_reward += reward
        self.UnityEnv.step()
        info = {}

        return self.state, reward, done, info

    def reset(self):
        self.UnityEnv.reset()
        self.state = self.initial_state
        self.time_length = 0
        print(f"Total Reward : {self.total_reward}")
        self.total_reward = 0
        return self.state


def change_env(current_env, new_env):
    current_env.UnityEnv.close()
    channel = EngineConfigurationChannel()
    new_UnityEnv = UnityEnvironment(file_name=new_env, side_channels=[channel])
    channel.set_configuration_parameters(time_scale=10000)
    new_UnityEnv.reset()
    road_num = int(new_env[new_env.index('Road') + 4])
    dest_list = np.array([[124.08, 0.16, 114.18],  # Position of destination in Road1
                          [118.58, 0.16, 9.61],  # Position of destination in Road2
                          [78.03, 0.16, 21.34]])  # Position of destination in Road3
    new_test_env = MyEnvironment(new_UnityEnv, dest_list[road_num - 1])  # Initialize MyEnvironment (OpenAI Gym)
    return new_test_env


# Set your file name here
# ==============================================================================================================
file_dir = 'Road3/Prototype 1'
road_num = int(file_dir[file_dir.index('Road') + 4])  # Get Road Number from str file_dir
# ==============================================================================================================

channel = EngineConfigurationChannel()
env = UnityEnvironment(file_name=file_dir, side_channels=[channel])
channel.set_configuration_parameters(time_scale=10000)
env.reset()
behavior_name = list(env.behavior_specs)[0]
decision_steps, terminal_steps = env.get_steps(behavior_name)
cur_obs = decision_steps.obs[0][0, :]
origin = cur_obs[0:3]

dest_list = np.array([[124.08, 0.16, 114.18],  # Position of destination in Road1
                      [118.58, 0.16, 9.61],  # Position of destination in Road2
                      [78.03, 0.16, 21.34]])  # Position of destination in Road3

test_env = MyEnvironment(env, dest_list[road_num - 1])  # Initialize MyEnvironment (OpenAI Gym)

# Settings
# =================================================================================================
training_mode = True
series_training_mode = True
series_training_timesteps_per_map = 100000
load_model = True
load_name = 'PPO_R123_35loops.zip'
save_name = 'Added_Callbacks'
# =================================================================================================

stop_callback = StopTrainingOnRewardThreshold(reward_threshold=3000, verbose=1) # Callbacks Settings

load_dir = os.path.join('Saved Models', load_name)
save_dir = os.path.join('Saved Models', save_name)
log_dir = os.path.join('Logs', save_name)

if training_mode:
    if series_training_mode:
        if load_model:
            model = PPO.load(load_dir, env=test_env, device='cpu', tensorboard_log=None)
            model.learn(total_timesteps=1)
            model.save('Series Training/PPO_Series_Training')
        else:
            model = PPO("MlpPolicy", test_env, verbose=1, device='cpu')
            model.learn(total_timesteps=1)
            model.save('Series Training/PPO_Series_Training')
        for i in range(20):
            test_env = change_env(test_env, 'Road1/Prototype 1')
            eval_callback = EvalCallback(test_env, callback_on_new_best=stop_callback, eval_freq=10000,
                                         best_model_save_path='Best Models/Road1', verbose=1)
            model = PPO.load('Series Training/PPO_Series_Training', env=test_env, device='cpu', tensorboard_log=os.path.join(log_dir, 'Road1'))
            model.learn(total_timesteps=series_training_timesteps_per_map, callback=eval_callback)
            model.save('Series Training/PPO_Series_Training')
            test_env = change_env(test_env, 'Road2/Prototype 1')
            eval_callback = EvalCallback(test_env, callback_on_new_best=stop_callback, eval_freq=10000,
                                         best_model_save_path='Best Models/Road2', verbose=1)
            model = PPO.load('Series Training/PPO_Series_Training', env=test_env, device='cpu', tensorboard_log=os.path.join(log_dir, 'Road2'))
            model.learn(total_timesteps=series_training_timesteps_per_map, callback=eval_callback)
            model.save('Series Training/PPO_Series_Training')
            test_env = change_env(test_env, 'Road3/Prototype 1')
            eval_callback = EvalCallback(test_env, callback_on_new_best=stop_callback, eval_freq=10000,
                                         best_model_save_path='Best Models/Road3', verbose=1)
            model = PPO.load('Series Training/PPO_Series_Training', env=test_env, device='cpu', tensorboard_log=os.path.join(log_dir, 'Road3'))
            model.learn(total_timesteps=series_training_timesteps_per_map, callback=eval_callback)
            model.save('Series Training/PPO_Series_Training')
        model.save(save_dir)
    else:
        if load_model:  # Continue training previous model
            model = PPO.load(load_dir, env=test_env, device='cpu', tensorboard_log=os.path.join(log_dir))
            model.learn(total_timesteps=2000000)
            model.save(save_dir)
        else:  # Create new model
            model = PPO("MlpPolicy", test_env, verbose=1, device='cpu', tensorboard_log=os.path.join(log_dir))
            model.learn(total_timesteps=2000000)
            model.save(save_dir)
else:  # Evaluate model
    model = PPO.load(load_dir, env=test_env, device='cpu')
    channel.set_configuration_parameters(time_scale=10)
    evaluate_policy(model, test_env, n_eval_episodes=10)

test_env.UnityEnv.close()
