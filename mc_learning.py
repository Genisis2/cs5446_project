import numpy as np
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
import pickle

from data import train_dataset, test_dataset

filename = 'linear_regression.sav'


def process_data(dataset):
    # rallies from different games merged into one list
    states, actions, rewards = [], [], []
    for data in dataset:
        for state, action, reward in zip(data['states'], data['actions'], data['rewards']):
            states.extend(state)
            actions.extend(action)
            rewards.extend(reward)

    inputs = []
    for state, action in zip(states, actions):
        inputs.append(np.concatenate([state, action]))
    inputs, rewards = np.array(inputs), np.array(rewards)
    return inputs, rewards


def train() -> None:
    inputs, rewards = process_data(train_dataset)

    model = LinearRegression()
    model.fit(inputs, rewards)

    # store the model
    pickle.dump(model, open(filename, 'wb'))


def test() -> None:
    inputs, rewards = process_data(test_dataset)
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(inputs, rewards)
    print(result)  # 0.047920123567392836, the result is not so good, I guess it might be caused by too much dimensions, not able to find the linear relationship


if __name__ == '__main__':
    train()
    test()
