import os
import gym
import yaml
import argparse
import numpy as np
import tensorflow as tf
from pprint import pprint
import matplotlib.pyplot as plt
from utils.resnet import ResNet18
from collections import OrderedDict
from stable_baselines.ppo2.ppo2 import constfn
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import VecNormalize
from utils.visualization import reachability_plot, plot2fig
from stable_baselines.deepq.policies import FeedForwardPolicy
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from utils import ALGOS, get_wrapper_class, linear_schedule, make_env

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

########################################################################################################################
# SET PATHS
# data path -> path to dataset | output path -> path to logging and model saving location
DATA_PATH = ""
OUTPUT_PATH = ""
#
########################################################################################################################
# SET ENVIRONMENT PARAMETERS
env_params = {'data_path': DATA_PATH,
              'verbose': 1,
              'test_patients': 5,
              'val_patients': 4,
              'prev_actions': 5,
              'prev_frames': 3,
              'val_set': np.array([25, 26, 27, 28]),
              'test_set': np.array([29, 30, 31, 32, 33]),
              'shuffles': 20,
              'chebishev': False,
              'no_nop': False,
              'max_time_steps': True,
              'time_step_limit': 50,
              'reward_goal_correct': 1.0,
              'reward_goal_incorrect': -0.25,
              'reward_move_closer': 0.05,
              'reward_move_further': -0.1,
              'reward_border_collision': -0.1}
# parameters to define the environment
#
########################################################################################################################

best_mean_reward, n_steps = -np.inf, 0

def custom_cnn(input, **kwargs):
    action_mem_size = 25
    action_history = input[:, 0, 0:action_mem_size, -1]
    input = input[..., :-1]

    action_values = ResNet18(x_input=input, classes=512)

    return action_values, action_history

class CustomCnnPolicy(FeedForwardPolicy):

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, obs_phs=None, dueling=True, **_kwargs):
        super(CustomCnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse, cnn_extractor=custom_cnn,
                                        feature_extraction="cnn", obs_phs=obs_phs, dueling=dueling, layer_norm=False, **_kwargs)

class Indicators():
    def __init__(self, num_states):
        self.num_states = num_states
        self.reset()

    def reset(self):
        self.reach_goal = []
        self.efficiency = []
        self.correct_actions = 0
        self.total_actions = 0

    def __str__(self):
        print("Goal reached {}% of the times".format(0 if not self.reach_goal else np.average(self.reach_goal)*100))
        print("Overall efficiency: {}%".format(np.average(0 if not self.efficiency else self.efficiency)*100))
        return ""


def callback(locals_, globals_):

    global n_steps, best_mean_reward
    self_       =   locals_.get('self')
    env_        =   self_.env.envs[0]
    info_       =   locals_.get('info')
    writer_     =   locals_.get('writer')
    episode_    =   locals_.get('num_episodes')

    # LOG CORRECT DECISIONS/TOTAL DECISIONS
    correct_decision_rate = info_.get('correct_decision_rate') if info_ else None
    if correct_decision_rate:
        summary = tf.Summary(value=[tf.Summary.Value(tag='callback_logging/correct_decision_rate', simple_value=correct_decision_rate)])
        writer_.add_summary(summary, episode_)

    episode_completed = locals_.get('done')
    if episode_completed and episode_ % 40 == 0:
        print("Done with episode {}".format(episode_))
        actions = []
        state_vals = []
        for state in range(env_.num_states):

            frame = env_.frames[env_.patient_idx][state][0]
            frames = np.repeat(frame[:, :, np.newaxis], len(env_.prev_frames), axis=2)
            observation = np.dstack((frames, np.zeros_like(frame)))

            action, q_vals, state_val = self_.predict(observation)
            actions.append(action)
            state_vals.append(state_val)

        # TEST CORRECTNESS ON TRAINING PATIENT
        quiver_plot, policy_correctness = env_.quiver_plot(states=list(range(env_.num_states)), actions=actions, state_vals=state_vals)
        summary = tf.Summary(value=[tf.Summary.Value(tag='callback_logging/policy correctness', simple_value=policy_correctness)])
        writer_.add_summary(summary, episode_)

        # LOG POLICY QUIVER PLOT
        if episode_ % 200 == 0:
            image = plot2fig(quiver_plot)
            summary = tf.Summary(value=[tf.Summary.Value(tag='Regular_training/Policy graph at episode {}'.format(episode_), image=image)])
            writer_.add_summary(summary, episode_)
        plt.close(quiver_plot)

    if episode_completed and episode_ % 100 == 0:
        avg_val_reachability = 0.0
        avg_val_correctness = 0.0
        max_time_steps = 20
        indicators = Indicators(env_.num_states)
        val_patients = env_.val_patient_idxs
        val_reachabilities = []

        for val_patient in val_patients:
            goals = env_.goals[val_patient]

            for state in range(env_.num_states):

                obs = env_.set(val_patient, state)
                prev_state = env_.state
                for step in range(max_time_steps):

                    if env_.no_nop and env_.state in goals:
                        done = True
                    else:
                        action, q_vals, state_val = model.predict(obs)
                        obs, reward, done, info = env_.step(action)

                        moving = not (prev_state == env_.state)
                        if moving:
                            indicators.total_actions += 1
                            if reward > 0:
                                indicators.correct_actions += 1

                    if done or step == max_time_steps - 1:
                        if (env_.state in goals and env_.no_nop) or (not env_.no_nop and reward == env_.reward_dict["goal_correct"]):
                            indicators.reach_goal.append(1)
                        else:
                            indicators.reach_goal.append(0)
                        _ = env_.reset()
                        break

                    prev_state = env_.state

            val_reachabilities.append(np.average(indicators.reach_goal))
#            avg_val_reachability += np.average(indicators.reach_goal)/len(val_patients) if isinstance(val_patients, (list, np.ndarray)) else np.average(indicators.reach_goal)
            print("Correctness for test patient {}: {}".format(val_patient, indicators.correct_actions / indicators.total_actions))
            print("Reachability for validation patient {}: {}".format(val_patient, np.average(indicators.reach_goal)))
            avg_val_correctness += indicators.correct_actions / indicators.total_actions / len(val_patients) if isinstance(val_patients, (list, np.ndarray)) \
                else indicators.correct_actions / indicators.total_actions
            indicators.reset()

        val_median_reachability = np.median(val_reachabilities)
        summary = tf.Summary(value=[tf.Summary.Value(tag='callback_logging/val_median_reachability', simple_value=val_median_reachability)])
        writer_.add_summary(summary, episode_)
        summary = tf.Summary(value=[tf.Summary.Value(tag='callback_logging/val_correctness', simple_value=avg_val_correctness)])
        writer_.add_summary(summary, episode_)

        #        if env_.val_reachability < avg_val_reachability:
        if env_.val_reachability < val_median_reachability:
            print("Improved the model at episode {}!".format(episode_))
            self_.save(OUTPUT_PATH + "val_model_episode_{}".format(episode_), cloudpickle=True)
            env_.val_reachability = val_median_reachability

    n_steps += 1

    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, nargs='+', default=["gym_sacrum_nav:sacrum_nav-v2"], help='environment ID(s)')
    parser.add_argument('-tb', '--tensorboard-log', help='Tensorboard log dir', default='./runs/', type=str)
    parser.add_argument('-i', '--trained-agent', help='Path to a pretrained agent to continue training', default='', type=str)
    parser.add_argument('--algo', help='RL Algorithm', default='dqn', type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument('-n', '--n-timesteps', help='Overwrite the number of timesteps', default=-1, type=int)
    parser.add_argument('--log-interval', help='Override log interval (default: -1, no change)', default=-1, type=int)
    parser.add_argument('-f', '--log-folder', help='Log folder', type=str, default='logs')
    parser.add_argument('--data-folder', help='Data folder', type=str, default='./data/')
    parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
    parser.add_argument('--n-trials', help='Number of trials for optimizing hyperparameters', type=int, default=10)
    parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO, 2: debug)', default=1, type=int)
    parser.add_argument('--gym-packages', type=str, nargs='+', default=[], help='Additional external Gym environemnt package modules to import (e.g. gym_minigrid)')
    args = parser.parse_args()

    env_params['data_path'] = args.data_folder

    env_ids = args.env
    set_global_seeds(args.seed)

    for env_id in env_ids:
        tensorboard_log = None if args.tensorboard_log == '' else os.path.join(args.tensorboard_log, env_id)
        print("=" * 10, env_id, "=" * 10)

        # Load hyperparameters from yaml file
        with open('hyperparams/{}.yml'.format(args.algo), 'r') as f:
            hyperparams_dict = yaml.full_load(f)
            if env_id in list(hyperparams_dict.keys()):
                hyperparams = hyperparams_dict[env_id]
            else:
                raise ValueError("Hyperparameters not found for {}-{}".format(args.algo, env_id))

        saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])
        algo_ = args.algo

        if args.verbose > 0:
            pprint(saved_hyperparams)

        n_envs = hyperparams.get('n_envs', 1)

        if args.verbose > 0:
            print("Using {} environments".format(n_envs))

        n_timesteps = int(hyperparams['n_timesteps'])

        # Delete keys so the dict can be pass to the model constructor
        if 'n_envs' in hyperparams.keys():
            del hyperparams['n_envs']
        del hyperparams['n_timesteps']

        env_wrapper = get_wrapper_class(hyperparams)
        if 'env_wrapper' in hyperparams.keys():
            del hyperparams['env_wrapper']

        if algo_ in ["ppo2"]:
            for key in ['learning_rate', 'cliprange', 'cliprange_vf']:
                if key not in hyperparams:
                    continue
                if isinstance(hyperparams[key], str):
                    schedule, initial_value = hyperparams[key].split('_')
                    initial_value = float(initial_value)
                    hyperparams[key] = linear_schedule(initial_value)
                elif isinstance(hyperparams[key], (float, int)):
                    # Negative value: ignore (ex: for clipping)
                    if hyperparams[key] < 0:
                        continue
                    hyperparams[key] = constfn(float(hyperparams[key]))
                else:
                    raise ValueError('Invalid value for {}: {}'.format(key, hyperparams[key]))
        normalize = False
        normalize_kwargs = {}
        if 'normalize' in hyperparams.keys():
            normalize = hyperparams['normalize']
            if isinstance(normalize, str):
                normalize_kwargs = eval(normalize)
                normalize = True
            del hyperparams['normalize']

        def create_env(env_params):
            global hyperparams

            if algo_ in ['dqn']:
                env = gym.make(env_id, env_params=env_params)
                env.seed(args.seed)
                if env_wrapper is not None:
                    env = env_wrapper(env)
            else:
                env = DummyVecEnv([make_env(env_id, 0, args.seed, wrapper_class=env_wrapper, env_params=env_params)])
                if normalize:
                    if args.verbose > 0:
                        if len(normalize_kwargs) > 0:
                            print("Normalization activated: {}".format(normalize_kwargs))
                        else:
                            print("Normalizing input and reward")
                    env = VecNormalize(env, **normalize_kwargs)
            return env

        env = create_env(env_params)

        env = DummyVecEnv([lambda: env])

    print(hyperparams)
    model = ALGOS[args.algo](CustomCnnPolicy,
                             env=env,
                             tensorboard_log=tensorboard_log,
                             verbose=args.verbose,
                             batch_size=64,
                             **hyperparams)
    print("Model loaded!")
    model.is_tb_set = False

    kwargs = {}
    if args.log_interval > -1:
        kwargs = {'log_interval': args.log_interval}

    model.learn(n_timesteps, callback=callback, **kwargs)

    model.save(OUTPUT_PATH + "final_model")
