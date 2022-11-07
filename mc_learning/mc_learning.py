import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
import os
from data import train_dataset, test_dataset

curr_dirpath = os.path.dirname(os.path.realpath(__file__))
model_filepath =  os.path.join(curr_dirpath, 'linear_regression.sav')

def process_data(dataset):
    # rallies from different games merged into one list
    inputs, accum_reward = [], []
    for data in dataset:
        for state, action, reward in zip(data['states'], data['actions'], data['rewards']):
            sa_pairs = np.concatenate((state, action), axis=-1)
            inputs.extend(sa_pairs)
            # target of each state in MC learning is observed total reward from state s onward
            # since we only have reward at the end of the trial, the reward for each state is just the end result
            # If winner
            if reward[-1] == 1:
                accum_reward.extend(np.ones(reward.shape))
            # If loser
            else:
                accum_reward.extend(np.zeros(reward.shape))
            
    inputs, accum_reward = np.array(inputs), np.array(accum_reward)
    return inputs, accum_reward


def train(save=True) -> LinearRegression:
    inputs, accum_reward = process_data(train_dataset)

    model = LinearRegression(fit_intercept=True)
    model.fit(inputs, accum_reward)

    # Store model if necessary
    if save:
        # store the model
        with open(model_filepath, 'wb') as fs:
            pickle.dump(model, fs)

    return model


def test() -> None:
    inputs, accum_reward = process_data(test_dataset)
    loaded_model = pickle.load(open(model_filepath, 'rb'))
    result = loaded_model.score(inputs, accum_reward)
    print(result)  # 0.047920123567392836, the result is not so good, I guess it might be caused by too much dimensions, not able to find the linear relationship


if __name__ == '__main__':
    train(False)
    test()
