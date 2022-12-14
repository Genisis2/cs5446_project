import os
import pandas as pd
import numpy as np

_DATA_DIRP = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'csv/')

for csv_fn in os.listdir(_DATA_DIRP):
    # ======================================================
    # Modify as needed to print out the data you want to see
    # ======================================================
    csv_fp = os.path.join(_DATA_DIRP, csv_fn)
    game_df = pd.read_csv(csv_fp)    

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
        if (is_final_shot == 1):
            # Get the sequence of strokes representing the rally
            rally_end = stroke_idx + 1
            rally = game_df[rally_start:rally_end]

            # If the last stroke is a loss for the hitter, 
            # assign reward to the prev stroke
            rally_length = rally_end - rally_start
            # If rally_length == 1 and no rewards, it means
            # the hitter failed at the serve stroke
            if (rally_length > 1
                    and rally.iloc[-1, reward_loc] != 1):
                rally.iloc[-2, reward_loc] = 1

            rallies.append(rally)
            # Reset rally start
            rally_start = rally_end

    # Extract features
    extracted_df = game_df[state_cols + action_cols + reward_cols]
    extracted_df.to_html(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'html/' + csv_fn + '.html'))