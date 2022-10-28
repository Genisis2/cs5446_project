from data import train_dataset, test_dataset

# Get the sequences of strokes for the train dataset's first match's first rally
first_match_first_rally_state_seq = train_dataset['states'][0][0]
first_match_first_rally_actions_seq = train_dataset['actions'][0][0]
first_match_first_rally_rewards_seq = train_dataset['rewards'][0][0]

print(first_match_first_rally_state_seq)
print(first_match_first_rally_actions_seq)
print(first_match_first_rally_rewards_seq)

# Get the first stroke (should be a serve) for the test dataset's 3rd match 6th rally
third_match_sixth_rally_state_serve = test_dataset['states'][2][5][0]
third_match_sixth_rally_actions_serve = test_dataset['actions'][2][5][0]
third_match_sixth_rally_rewards_serve = test_dataset['rewards'][2][5][0]

print(third_match_sixth_rally_state_serve)
print(third_match_sixth_rally_actions_serve)
print(third_match_sixth_rally_rewards_serve)

train_states = train_dataset['states']
train_actions = train_dataset['actions']
train_rewards = train_dataset['rewards']

test_states = test_dataset['states']
test_actions = test_dataset['actions']
test_rewards = test_dataset['rewards']

# Sanity check that all rows in all files are included
total_rows_per_match = []
for x in range(3):
    count = 0
    for y in range(len(train_states[x])):
        # All columns should add up to 13 states + 5 actions + 1 reward
        total_columns = train_states[x][y].shape[1] + \
            train_actions[x][y].shape[1] + train_rewards[x][y].shape[1]
        assert 19 == total_columns
        # States/actions/rewards should all have the same number of rows
        assert train_states[x][y].shape[0] == train_actions[x][y].shape[0] == train_rewards[x][y].shape[0]
        count += train_states[x][y].shape[0]

    for y in range(len(test_states[x])):
        # All columns should add up to 13 states + 5 actions + 1 reward
        total_columns = test_states[x][y].shape[1] + \
            test_actions[x][y].shape[1] + test_rewards[x][y].shape[1]
        assert 19 == total_columns
        # States/actions/rewards should all have the same number of rows
        assert test_states[x][y].shape[0] == test_actions[x][y].shape[0] == test_rewards[x][y].shape[0]
        count += test_states[x][y].shape[0]
    total_rows_per_match.append(count)

print(
    f"First match total rows: {total_rows_per_match[0]}: {total_rows_per_match[0] == 417}")
print(
    f"Second match total rows: {total_rows_per_match[1]}: {total_rows_per_match[1] == 1234}")
print(
    f"Third match total rows: {total_rows_per_match[2]}: {total_rows_per_match[2] == 463}")
