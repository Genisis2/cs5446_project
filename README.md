# Tennis Action Evaluator

## Reference implementation for Deep Reinforcement Learning
https://github.com/Guiliang/DRL-ice-hockey

# Q-definition
- Let $Q(s,a)$ = Probability of the `hit_id` player at state $s$ winning the rally with action $a$
- Loss function for DLR
    - $L_t(\theta_t) = \mathop{\mathbb{E}}[(g_t + (1 - Q(S_{t+1}, a_{t+1}, \theta_{t})) - Q(S_{t}, a_{t}, \theta_{t})]$
        - $Q(S_{t}, a_{t}, \theta_{t})$: Probability of `hit_id` player at $s_t$ winning the rally
        - $1 - Q(S_{t+1}, a_{t+1}, \theta_{t})$: Still the probability of `hit_id` at $s_t$ winning the game because $Q(S_{t+1}, a_{t+1}, \theta_{t})$ is the just probability that `opp_id` wins the game. 

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
- `is_player_one`: (need to add) Indicates if the current `hit_id` is player one, where player one is the player in the game with the lower ID number.

#### Actions
- `curr_hit_x`: (Check later if these features affect results) X-pos of the hitting player at the moment of hitting the ball
- `curr_hit_y`: (Check later if these features affect results) Y-pos of the hitting player at the moment of hitting the ball
- `shot_type`: (need to add): 0, `is_serve`. 1, `is_forehand`. 2, `is_backhand`
- `bounce_x`: X-pos where the ball bounces on the court after `hit_id` hits back
- `bounce_y`: Y-pos where the ball bounces on the court after `hit_id` hits back

#### Goal
- `reward` (need to add): 1 if `final_outcome` and `is_final_shot` are both 1. 0, otherwise.

#### Unused columns
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

# Options for implementation
- Follow the model described in the network and use SARSA update term as a loss function
- Implement some sort of direct utility estimation since we have several rallies we can use (MC learning)
    - If an action leads to 1, we train to 1. If action leads to 0, we train to 0.
- (Impossible) Follow the model described in the network and use Q-learning update term as a loss function
    - Why impossible? Q-learning requires choosing an action $a'$ that maximizes the Q-function at $s'$. But in our case, we are dealing with continuous values in our definition for actions. Can't pick an $a'$.
    - Pros: Can compare the 2 and see which performs better