from data import states, actions, rewards

# Get the sequences of strokes for the first match's first rally
first_match_first_rally_state_seq = states[0][0]
first_match_first_rally_actions_seq = actions[0][0]
first_match_first_rally_rewards_seq = rewards[0][0]

print(first_match_first_rally_state_seq)
print(first_match_first_rally_actions_seq) 
print(first_match_first_rally_rewards_seq)

# Get the first stroke (should be a serve) for 3rd match 6th rally
third_match_sixth_rally_state_serve = states[2][5][0]
third_match_sixth_rally_actions_serve = actions[2][5][0]
third_match_sixth_rally_rewards_serve = rewards[2][5][0]

print(third_match_sixth_rally_state_serve)
print(third_match_sixth_rally_actions_serve) 
print(third_match_sixth_rally_rewards_serve)

# Sanity check that all rows in all files are included
total_rows_per_match = []
for x in range(len(states)):
    count = 0
    for y in range(len(states[x])):
        # All columns should add up to 12 states + 5 actions + 1 reward
        total_columns = states[x][y].shape[1] + actions[x][y].shape[1] + rewards[x][y].shape[1]
        assert 18 == total_columns
        # States/actions/rewards should all have the same number of rows
        assert states[x][y].shape[0] == actions[x][y].shape[0] == rewards[x][y].shape[0]
        count += states[x][y].shape[0]
    total_rows_per_match.append(count)

print(f"First match total rows: {total_rows_per_match[0]}: {total_rows_per_match[0] == 417}")
print(f"Second match total rows: {total_rows_per_match[1]}: {total_rows_per_match[1] == 1234}")
print(f"Third match total rows: {total_rows_per_match[2]}: {total_rows_per_match[2] == 463}")