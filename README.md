# Tennis Action Evaluator

## Final experiment data
- [final_experiment.ipynb](final_experiment.ipynb)

## Directory structure
- `d_sarsa`: Code for NN model for MC learning and SARSA Learning
- `data`: Raw data and code for data processing
- `evaluation`: Code for scoring models
- `experiment_models`: Contains models from different experiment runs
- `mc_learning`: Code for linear regression MC learning
- `report_archive`: Jupyter notebooks of archived experiments

## Reference implementation for Deep Reinforcement Learning
- https://github.com/Guiliang/DRL-ice-hockey

## Q-definition
- Let $Q(s,a)$ = Probability of the agent at state $s$ winning the rally with action $a$

## Dataset
Found in `data` directory.

### Vocabulary
https://en.wikipedia.org/wiki/Glossary_of_tennis_terms

- `match`
    - Each `.csv` file is a match
    - Each match consists of several `rallies`
- `rally`
    - A sequence of `strokes` that ends when someone gets a point 
- `stroke`
    - Each row in the `.csv` file represents a stroke

### Columns

#### States
- `prev_hit_x`: X-pos of the hitting player at the moment of hitting the ball at the prev turn
- `prev_hit_y`: Y-pos of the hitting player at the moment of hitting the ball at the prev turn
- `curr_hit_x`: X-pos of the hitting player at the moment of hitting the ball
- `curr_hit_y`: Y-pos of the hitting player at the moment of hitting the ball
- `prev_opp_x`: X-pos of the other player at the moment of hitting the ball at the prev turn
- `prev_opp_y`: Y-pos of the other player at the moment of hitting the ball at the prev turn
- `hit_move_dist`
- `hit_move_speed` 
- `curr_opp_x`: X-pos of the other player at the moment of hitting the ball
- `curr_opp_y`: Y-pos of the other player at the moment of hitting the ball
- `opp_move_dist`: Pre-calculated feature of distance of `prev_opp_x/y` and `curr_opp_x/y`
- `opp_move_speed`: Pre-calculated feature of speed
- `prev_bounce_x`: X-pos of the incoming bounce
- `prev_bounce_y`: Y-pos of the incoming bounce

#### Actions
- `shot_type`: (need to add): -1 not hitting, 0, `is_serve`. 1, `is_forehand`. 2, `is_backhand`
- `bounce_x`: 0 if `shot_type` is -1. X-pos where the ball bounces on the court after `hit_id` hits back
- `bounce_y`: 0 if `shot_type` is -1. Y-pos where the ball bounces on the court after `hit_id` hits back

#### Goal
- `reward` (need to add): 1 if `final_outcome` and `is_final_shot` are both 1. 0, otherwise.

#### Other columns
- `hit_id`: Identifier for the hitting player in the current stroke
- `opp_id`: Identifier for the other player in the current stroke
- `match_id`: Identifier for the match
- `court_type`: Identifier for the court type
- `time_ms`: The current game time
- `time_diff`: The difference in game time from the current stroke and the previous stroke
- `is_far`: "Far player" describes the player that is far in the camera's perspective
- `prev_bounce_time_ms`: Game time when incoming ball bounces
- `bounce_time_ms`: Game time when the ball bounces on the court after `hit_id` hits back
- `outcome`: Outcome of the current shot. 1 if ball successfully crosses the net and is within bounds. 0, otherwise. 
- `final_outcome`: 1 if `hit_id` wins the rally. 0, otherwise
- `is_final_shot`: 1 if last shot of the rally. 0, otherwise