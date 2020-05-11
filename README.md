# Ultrasound-Guided Robotic Navigation with Deep Reinforcement Learning


Code for: 
```
@misc{hase2020ultrasoundguided,
    title={Ultrasound-Guided Robotic Navigation with Deep Reinforcement Learning},
    author={Hannes Hase and Mohammad Farid Azampour and Maria Tirindelli and Magdalini Paschali and Walter Simson and Emad Fatemizadeh and Nassir Navab},
    year={2020},
    eprint={2003.13321},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
# The project
This project aims at learning a policy for autonomously navigating to the sacrum in simulated lower back environments from volunteers. As for the deep reinforcement learning agent, we use a double dueling DQN with a prioritized replay memory. 

For the implementation of this project, we used the [rl-zoo](https://github.com/araffin/rl-baselines-zoo) framework, a slightly adapted [stable-baselines](https://github.com/hhase/stable-baselines) library and an [environment](https://github.com/hhase/gym_sacrum_env) built using the [gym](https://gym.openai.com/) toolkit. 

# Setup
To run the code, first, some parameters need to be set.

 - `DATA_PATH`: corresponds to the location of the [dataset](https://github.com/hhase/sacrum_data-set).
 - `OUTPUT_PATH`: corresponds to the path for the output.
 - `test_patients`: amount of test environments.
 - `val_patients`: amount of validation environments.
 - `prev_actions`: size of the action memory.
 - `prev_frames`: size of the previous frame memory.
 - `val_set`: if defined, sets the environments to be used for validation.
 - `test_set`: if defined, sets the environments to be used for testing.
 - `shuffles`: amount of random shuffles for train/val/test set creation. Only relevant if test and validation sets are not defined.
 - `chebishev`: boolean that enables diagonal movements.
 - `no_nop`: boolean that removes the stopping action from the action space. Used for MS-DQN architecture.
 - `max_time_steps`: boolean that enables resetting the agent when it takes too long to reach a goal state.
 - `time_step_limit`: the amount of time steps the agent has to reach a goal state.
