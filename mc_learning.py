import numpy as np
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
import pickle

from data import train_dataset, test_dataset

filename = 'finalized_model.sav'

def _flatten(array):
    return [item for sublist in array for item in sublist]

def train() -> None:
    states, actions, rewards = train_dataset['states'], train_dataset['actions'], train_dataset['rewards']
    
    # flatten the game index - rallies from different games merged into one list
    states, actions, rewards = _flatten(states), _flatten(actions), _flatten(rewards)
    
    model = LinearRegression()
    model.fit(zip(states, actions), rewards)
    print(model.score(zip(states, actions), rewards))

    # store the model
    pickle.dump(model, open(filename, 'wb'))


def test() -> None:
    states, actions, rewards = test_dataset['states'], test_dataset['actions'], test_dataset['rewards']
    states, actions, rewards = _flatten(states), _flatten(actions), _flatten(rewards)
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(zip(states, actions), rewards)
    print(result)


if __name__ == '__main__':
    train()
