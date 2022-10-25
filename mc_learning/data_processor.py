import os
import pandas as pd

DATA_1_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data\\csv\\m-001-data-v3.csv')
DATA_7_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data\\csv\\m-007-data-v3.csv')
DATA_99_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data\\csv\\m-099-data-v3.csv')

DF_1 = pd.read_csv(DATA_1_PATH)
DF_2 = pd.read_csv(DATA_7_PATH)
DF_3 = pd.read_csv(DATA_99_PATH)

df = pd.concat([DF_1, DF_2, DF_3], ignore_index=True)

DATA_COLS = [
    'hit_id',
    'opp_id',
    'is_far',
    'prev_hit_x',
    'prev_hit_y',
    'prev_opp_x',
    'prev_opp_y',
    'curr_hit_x',
    'curr_hit_y',
    'curr_opp_x',
    'curr_opp_y',
    'hit_move_dist',
    'hit_move_speed',
    'opp_move_dist',
    'opp_move_speed',
    'is_serve',
    'is_forehand',
    'is_backhand',
    'outcome',
    'final_outcome',
    'prev_bounce_time_ms',
    'prev_bounce_x',
    'prev_bounce_y',
    'bounce_time_ms',
    'bounce_x',
    'bounce_y',
    'is_final_shot'
]

STATE_COLS = [
    'prev_hit_x',
    'prev_hit_y',
    'prev_opp_x',
    'prev_opp_y',
    'hit_move_dist',
    'hit_move_speed',
    'curr_opp_x',
    'curr_opp_y',
    'opp_move_dist',
    'opp_move_speed',
    'prev_bounce_x',
    'prev_bounce_y'
]

ACTION_COLS = [
    'curr_hit_x',
    'curr_hit_y', 
    'is_serve',
    'is_forehand',
    'is_backhand',
    'bounce_x',
    'bounce_y'
]

REWARD_COLS = [
    'reward'
]

def split_trials():
    start = 0
    trials = []
    for i in range(df.shape[0]):
        if df.loc[[i]]['is_final_shot'].item() == 1:
            arr = df[DATA_COLS].loc[start:i].to_numpy()
            trial_odd_row = arr[::2,:]
            trial_even_row = arr[1::2,:]
            trial = [trial_odd_row]
            if(len(trial_even_row)>0):
                trial.append(trial_even_row)
            trials.append(trial)
            start = i+1
    return trials



# def _initialize_data() -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]], List[List[np.ndarray]]]:
#     """Initializes tennis match data using CSV's in csv/

#     Returns
#         Tuple(states, actions, rewards):
#                 2D Lists of `np.ndarray`'s describing the `states`, `actions`, and respective `rewards`. 
#     """
#     # Features to extract
#     state_cols = [
#         'prev_hit_x',
#         'prev_hit_y',
#         'prev_opp_x',
#         'prev_opp_y',
#         'hit_move_dist',
#         'hit_move_speed',
#         'curr_opp_x',
#         'curr_opp_y',
#         'opp_move_dist',
#         'opp_move_speed',
#         'prev_bounce_x',
#         'prev_bounce_y'
#     ]
#     action_cols = [
#         'curr_hit_x',
#         'curr_hit_y', 
#         'shot_type',
#         'bounce_x',
#         'bounce_y'
#     ]
#     reward_cols = [
#         'reward'
#     ]
    
#     # List of matches holding list of rallies holding a 2D array of strokes 
#     states:List[List[np.ndarray]] = []
#     actions:List[List[np.ndarray]] = []
#     rewards:List[List[np.ndarray]] = []

#     # Read the data from the input files
#     for csv_fn in os.listdir(_DATA_DIRP):
#         csv_fp = os.path.join(_DATA_DIRP, csv_fn)
#         game_df = pd.read_csv(csv_fp)

#         # Add in shot_type column
#         one_hot_shot_type = game_df[['is_serve', 'is_forehand', 'is_backhand']].to_numpy()
#         shot_type_val = np.argwhere(one_hot_shot_type == 1)[:,1] # Get only values of second col
#         game_df['shot_type'] = shot_type_val

#         # Add in reward
#         # 1 if both 1, else 0
#         game_df['reward'] = game_df['final_outcome'].to_numpy() * game_df['is_final_shot'].to_numpy()

#         # Extract features
#         state_df = game_df[state_cols]
#         action_df = game_df[action_cols]
#         reward_df = game_df[reward_cols]

#         # Split into rallies for feeding into the LSTM network.
#         state_rallies:List[np.ndarray] = []
#         action_rallies:List[np.ndarray] = []
#         reward_rallies:List[np.ndarray] = []
#         rally_start = 0
#         for stroke_idx, is_final_shot in game_df['is_final_shot'].items():
#             # If final shot of the rally
#             if (is_final_shot == 1):
#                 rally_end = stroke_idx + 1
#                 state_rallies.append(state_df[rally_start:rally_end].to_numpy())
#                 action_rallies.append(action_df[rally_start:rally_end].to_numpy())
#                 reward_rallies.append(reward_df[rally_start:rally_end].to_numpy())
#                 # Reset rally start
#                 rally_start = rally_end

#         # Add rallies to match lists
#         states.append(state_rallies)
#         actions.append(action_rallies)
#         rewards.append(reward_rallies)

#     return states, actions, rewards

# states:List[List[np.ndarray]]
# """
# 2D list of 2D `np.ndarray`'s representing the states within the data.

# Note: states[0][1][3] will return the `np.ndarray` representing the state at match 0, rally 1, stroke 2.
# """
# actions:List[List[np.ndarray]]
# """
# 2D list of 2D `np.ndarray`'s representing the actions within the data.

# Note: actions[0][1][3] will return the `np.ndarray` representing the action at match 0, rally 1, stroke 2.
# """
# rewards:List[List[np.ndarray]]
# """
# 2D list of 2D `np.ndarray`'s representing the rewards within the data.

# Note: rewards[0][1][3] will return the `np.ndarray` representing the reward at match 0, rally 1, stroke 2.
# """

# states, actions, rewards = _initialize_data()
