from lib2to3.pytree import Base
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
from math import gamma
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results, plot_curves

from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback, 
    CheckpointCallback, 
    StopTrainingOnMaxEpisodes
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed


from nn.ppo_nn import CustomActorCriticPolicy
from gym_env import RBCEnv



# Create log dir
log_dir = "train_log"
model_dir = 'models/self_play_models'
tensorboard_dir = os.path.join(log_dir, "self_play_model_tensorboard")
checkpoint_dir = 'checkpoints'

def make_env(model_dir, rank, num_episodes, seed=0, ):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = RBCEnv(model_dir=model_dir)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

def callback_on_new_best(): 
    print('NEW BEST MODEL SAVED')


class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """
    def __init__(self, pbar, max_episodes, verbose = 0):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar
        self.max_episodes = max_episodes
        self._total_max_episodes = max_episodes
        self.n_episodes = 0

    def _init_callback(self) -> None:
        # At start set total max according to number of envirnments
        self._total_max_episodes = self.max_episodes * self.training_env.num_envs

    def _on_step(self):
        # Update the progress bar:
        done_array = np.array(self.locals.get("done") if self.locals.get("done") is not None else self.locals.get("dones"))
        self.n_episodes += np.sum(done_array).item()
        self._pbar.n = self.n_episodes
        self._pbar.update(0)

# this callback uses the 'with' block, allowing for correct initialisation and destruction
class ProgressBarManager(object):
    def __init__(self, total_max_episodes): # init object with total timesteps
        self.pbar = None
        self.total_max_episodes = total_max_episodes
        
    def __enter__(self): # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_max_episodes)
            
        return ProgressBarCallback(self.pbar, self.total_max_episodes)

    def __exit__(self, exc_type, exc_val, exc_tb): # close the callback
        self.pbar.n = self.total_max_episodes
        self.pbar.update(0)
        self.pbar.close()


if __name__ == '__main__':
    # n_env = 1
    total_episodes = 10000
    # env = SubprocVecEnv([make_env(model_dir, i, 0) for i in range(n_env)])
    env = RBCEnv(model_dir=model_dir)
    env = Monitor(env, log_dir)
    eval_env = RBCEnv(model_dir=model_dir)
    eval_env = Monitor(eval_env, log_dir)

    # model = PPO.load(os.path.join(model_dir, "best_model_gamma05"))
    model = PPO(CustomActorCriticPolicy, env, verbose = 1, tensorboard_log=tensorboard_dir)
    model.save(os.path.join(model_dir, "self_play_best_model")) 

    # checkpoint_callback = CheckpointCallback(
    # save_freq=10,
    # save_path=checkpoint_dir,
    # name_prefix="rl_model"
    # ) 

    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=model_dir, 
        callback_on_new_best=callback_on_new_best(),
        log_path=log_dir, 
        eval_freq= 5000,
        n_eval_episodes=5, 
        deterministic= True, 
        render=False)

    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=total_episodes, verbose=1)

    with ProgressBarManager(total_episodes) as tqdm_callback: 
        model.learn(int(1e10), callback= [callback_max_episodes, eval_callback, tqdm_callback], tb_log_name="self_play_model")
    model.save(os.path.join(model_dir, "final_self_play_model"))
    plot_results([log_dir],int(1e10), results_plotter.X_EPISODES, "PPO RBC")
    # plot_curves([log_dir], results_plotter.X_EPISODES, "PPO RBC")
    plt.show()
