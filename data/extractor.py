"""
Extracts and processes the data from the csv into states, actions, rewards 
"""

import os
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Constants
_DATA_DIRP = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'csv/')
_TRAIN_PERCENT = 0.8
_SAMPLE_SEED = 42

def _initialize_data() -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]], List[List[np.ndarray]]]:
    """Initializes tennis match data using CSV's in csv/

    Returns
        Tuple(states, actions, rewards):
                2D Lists of `np.ndarray`'s describing the `states`, `actions`, and respective `rewards`. 
    """
    # Features to extract
    state_cols = [
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
        'prev_bounce_y',
        'is_player_one'
    ]
    action_cols = [
        'curr_hit_x',
        'curr_hit_y', 
        'shot_type',
        'bounce_x',
        'bounce_y'
    ]
    reward_cols = [
        'reward'
    ]
    
    train_dataset:Dict[str, List[List[np.ndarray]]] = {
      'states': [],
      'actions': [],
      'rewards': []
    }
    test_dataset:Dict[str, List[List[np.ndarray]]] = {
      'states': [],
      'actions': [],
      'rewards': []
    }

    # Read the data from the input files
    for csv_fn in os.listdir(_DATA_DIRP):
        csv_fp = os.path.join(_DATA_DIRP, csv_fn)
        game_df = pd.read_csv(csv_fp)

        # Add in shot_type column
        one_hot_shot_type = game_df[['is_serve', 'is_forehand', 'is_backhand']].to_numpy()
        shot_type_val = np.argwhere(one_hot_shot_type == 1)[:,1] # Get only values of second col
        game_df['shot_type'] = shot_type_val

        # Add in is_player_one column
        player_one_id = np.min(game_df[['hit_id', 'opp_id']].to_numpy())
        game_df['is_player_one'] = np.where(game_df['hit_id'] == player_one_id, 1, 0)

        # Add in reward
        # 1 if both 1, else 0
        game_df['reward'] = game_df['final_outcome'].to_numpy() * game_df['is_final_shot'].to_numpy()

        # Split into rallies for feeding into the LSTM network.
        reward_loc = game_df.columns.get_loc('reward')
        rallies = []
        rally_start = 0
        for stroke_idx, is_final_shot in game_df['is_final_shot'].items():
            # If final shot of the rally
            if is_final_shot == 1:
                # Get the sequence of strokes representing the rally
                rally_end = stroke_idx + 1
                rally = game_df[rally_start:rally_end]

                # If the last stroke is a loss for the hitter, 
                # assign reward to the prev stroke
                rally_length = rally_end - rally_start
                # If rally_length == 1 and no rewards, it means
                # the hitter failed at the serve stroke
                if (rally_length > 1 and rally.iloc[-1, reward_loc] != 1):
                    rally.iloc[-2, reward_loc] = 1

                rallies.append(rally)
                # Reset rally start
                rally_start = rally_end

        # Split rallies into train/test
        total_rallies = len(rallies)
        train_count = int(_TRAIN_PERCENT * total_rallies)
        rallies_train, rallies_test = train_test_split(
            rallies, 
            train_size=train_count, 
            random_state=_SAMPLE_SEED, 
            shuffle=True,
        )

        # Extract features
        train_states = []
        train_actions = []
        train_rewards = []
        test_states = []
        test_actions = []
        test_rewards = []
        for rally_train in rallies_train:
            train_states.append(rally_train[state_cols].to_numpy())
            train_actions.append(rally_train[action_cols].to_numpy())
            train_rewards.append(rally_train[reward_cols].to_numpy())
        for rally_test in rallies_test:
            test_states.append(rally_test[state_cols].to_numpy())
            test_actions.append(rally_test[action_cols].to_numpy())
            test_rewards.append(rally_test[reward_cols].to_numpy())

        # Store
        train_dataset['states'].append(train_states)
        train_dataset['actions'].append(train_actions)
        train_dataset['rewards'].append(train_rewards)
        test_dataset['states'].append(test_states)
        test_dataset['actions'].append(test_actions)
        test_dataset['rewards'].append(test_rewards)

    return train_dataset, test_dataset

# Split into train and test
train_dataset:Dict[str, List[List[np.ndarray]]] = {}
"""
Contains the training rally samples for states/actions/rewards

Keys: ''

Note: `train[states/actions/rewards][a][b][c][d]`
- `a` indicates the game index
- `b` indicates the rally index
- `c` indicates the stroke index
- `d` indicates the feature index
"""
test_dataset:Dict[str, List[List[np.ndarray]]] = {}
"""
Contains the testing rally samples for states/actions/rewards

Note: `test[states/actions/rewards][a][b][c][d]`
- `a` indicates the game index
- `b` indicates the rally index
- `c` indicates the stroke index
- `d` indicates the feature index
"""
train_dataset, test_dataset = _initialize_data()

