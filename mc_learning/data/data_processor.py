import os
import pandas as pd

DATA_1_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'csv/m-001-data-v3.csv')
DATA_7_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'csv/m-007-data-v3.csv')
DATA_99_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'csv/m-099-data-v3.csv')

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

# return a 2-d list
# each element is a numpy array for a trial
def _split_trials():
    start = 0
    trials = []
    for i in range(df.shape[0]):
        if df.loc[[i]]['is_final_shot'].item() == 1:

            # every step -0.4, last step +1, calculate overall reward for one trail
            arr = df[DATA_COLS].loc[start:i].to_numpy()
            trial_odd_row = arr[::2,:]
            trial_even_row = arr[1::2,:]
            trial = [trial_odd_row]
            if(len(trial_even_row)>0):
                trial.append(trial_even_row)
            trials.append(trial)
            start = i+1
    return trials

trails = _split_trials()