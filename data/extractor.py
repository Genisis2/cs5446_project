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

# Features to extract
hit_state_cols = [
    'prev_hit_x',
    'prev_hit_y',
    'curr_hit_x',
    'curr_hit_y', 
    'hit_move_dist',
    'hit_move_speed',
    'prev_opp_x',
    'prev_opp_y',
    'curr_opp_x',
    'curr_opp_y',
    'opp_move_dist',
    'opp_move_speed',
    'prev_bounce_x',
    'prev_bounce_y'
]
hit_action_cols = [
    'shot_type',
    'bounce_x',
    'bounce_y'
]
opp_state_cols = [
    'prev_opp_x',
    'prev_opp_y',
    'curr_opp_x',
    'curr_opp_y',
    'opp_move_dist',
    'opp_move_speed',
    'prev_hit_x',
    'prev_hit_y',
    'curr_hit_x',
    'curr_hit_y',
    'hit_move_dist',
    'hit_move_speed',
    'prev_bounce_x',
    'prev_bounce_y'
]
opp_action_cols = [
    'shot_type',
    'bounce_x',
    'bounce_y'
]

def _initialize_data() -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]]:
    """Initializes tennis match data using CSV's in csv/

    Returns
        train_dataset, test_dataset
    """
    train_dataset:List[Dict[str, np.ndarray]] = []
    test_dataset:List[Dict[str, np.ndarray]] = []

    # Read the data from the input files
    for csv_fn in os.listdir(_DATA_DIRP):
        csv_fp = os.path.join(_DATA_DIRP, csv_fn)
        game_df = pd.read_csv(csv_fp)

        # Add in shot_type column
        one_hot_shot_type = game_df[['is_serve', 'is_forehand', 'is_backhand']].to_numpy()
        shot_type_val = np.argwhere(one_hot_shot_type == 1)[:,1] # Get only values of second col
        game_df['shot_type'] = shot_type_val

        # Split into rallies for feeding into the LSTM network.
        rallies:List[pd.DataFrame] = []
        rally_start = 0
        for stroke_idx, is_final_shot in game_df['is_final_shot'].items():
            # If final shot of the rally
            if is_final_shot == 1:
                # Get the sequence of strokes representing the rally
                rally_end = stroke_idx + 1
                rally = game_df[rally_start:rally_end]
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

        def _extract_seqs_from_rally(rally:pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Extract sequences of states, action, reward triplets from the given rally `pd.DataFrame`
            
            Returns
                state_seq, action_seq, reward_seq 
            """

            stroke_count = rally.shape[0]
            # hit_x_cols or opp_x_cols doesn't matter since they have the same length
            state_col_count = len(hit_state_cols)
            action_col_count = len(hit_action_cols)

            # shape = (# of players, # of strokes in rally, # of features)
            # Let player at index 0 be winner. 1 is loser.
            state_seq = np.zeros((2, stroke_count, state_col_count))
            action_seq = np.zeros((2, stroke_count, action_col_count))
            # All rewards are zero until decided otherwise
            reward_seq = np.zeros((2, stroke_count, 1))

            rally_start_idx = rally.index.start
            for stroke_idx, stroke in rally.iterrows():

                # Normalize idx
                stroke_idx = stroke_idx - rally_start_idx

                # Get the hitter state/action/reward sequences
                hit_state = stroke[hit_state_cols].to_numpy()
                hit_action = stroke[hit_action_cols].to_numpy()

                # Get the opponent state/action/reward sequences
                opp_state = stroke[opp_state_cols].to_numpy()
                opp_action = stroke[opp_action_cols].to_numpy()
                opp_action[-3] = -1 # shot_type
                opp_action[-2] = 0 # bounce_x
                opp_action[-1] = 0 # bounce_y

                # Add the hitter/opponent df's to the winner or loser sequence, depending
                # on which player wins/loses
                if (stroke['final_outcome'] == 1):
                    # Hitter wins
                    state_seq[0][stroke_idx] = hit_state
                    action_seq[0][stroke_idx] = hit_action
                    state_seq[1][stroke_idx] = opp_state
                    action_seq[1][stroke_idx] = opp_action
                else:
                    # Hitter loses
                    state_seq[0][stroke_idx] = opp_state
                    action_seq[0][stroke_idx] = opp_action
                    state_seq[1][stroke_idx] = hit_state
                    action_seq[1][stroke_idx] = hit_action

                # If stroke is final shot, set the winner's reward as 1
                # Leave loser's reward as 0
                if (stroke['is_final_shot'] == 1):
                    reward_seq[0][stroke_idx] = 1

            return state_seq, action_seq, reward_seq

        # Extract features
        for rally_train in rallies_train:
            rally_state_seq, rally_action_seq, rally_reward_seq = _extract_seqs_from_rally(rally_train)
            rally_dict = {
                'states': rally_state_seq,
                'actions': rally_action_seq,
                'rewards': rally_reward_seq
            }
            train_dataset.append(rally_dict)
        for rally_test in rallies_test:
            # Extract data from each rally
            rally_state_seq, rally_action_seq, rally_reward_seq = _extract_seqs_from_rally(rally_test)
            rally_dict = {
                'states': rally_state_seq,
                'actions': rally_action_seq,
                'rewards': rally_reward_seq
            }
            test_dataset.append(rally_dict)

    return train_dataset, test_dataset

train_dataset:List[Dict[str, np.ndarray]]
"""
Contains the training rally samples for states/actions/rewards

The numpy array in this structure is a 3D array

Note: `train[a][states/actions/rewards][b][c][d]`
- `a` indicates the rally index
- `b` indicates the player index (0 for winner, 1 for loser)
- `c` indicates the stroke index
- `d` indicates the feature index
"""
test_dataset:List[Dict[str, np.ndarray]]
"""
Contains the testing rally samples for states/actions/rewards

The numpy array in this structure is a 3D array

Note: `test[a][states/actions/rewards][b][c][d]`
- `a` indicates the rally index
- `b` indicates the player index (0 for winner, 1 for loser)
- `c` indicates the stroke index
- `d` indicates the feature index
"""
train_dataset, test_dataset = _initialize_data()

