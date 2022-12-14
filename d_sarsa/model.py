from torch_snippets import Report
from typing import Any, Dict, Tuple
from data import train_dataset, test_dataset, hit_action_cols, hit_state_cols
import torch.nn as nn
import torch.optim as optim
import torch
import os
import numpy as np
import random
import evaluation as eval

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
curr_dirpath = os.path.dirname(os.path.realpath(__file__))
model_dirpath =  os.path.join(curr_dirpath, 'model/')
model_filepath =  os.path.join(model_dirpath, 'model.pth')

# Hyperparameters
learning_rate = 1e-3
gamma = 1
num_epochs = 200 # 30

# Architecture
lstm_input_size = len(hit_action_cols) + len(hit_state_cols) # Length of hit/opp are the same
lstm_hidden_size = 512
linear1_output = 1024
linear2_output = 1000
linear3_output = 1

# Model
class TennisEvalNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(lstm_input_size, lstm_hidden_size, batch_first=True)
        self.linear1 = nn.Linear(lstm_hidden_size, linear1_output)
        self.linear2 = nn.Linear(linear1_output, linear2_output)
        self.linear3 = nn.Linear(linear2_output, linear3_output)
        self.relu = nn.ReLU()

    def forward(self, sa_pairs):
        # Note: batch_size should be 2, one seq for each player in a rally
        # shape: (batch_size, seq_length, lstm_hidden_size)
        output, _ = self.lstm(sa_pairs)
        # shape: (batch_size, seq_length, linear1_output)
        output = self.linear1(output)
        output = self.relu(output)
        # shape: (batch_size, seq_length, linear2_output)
        output = self.linear2(output)
        output = self.relu(output)
        # shape: (batch_size, seq_length, linear3_output)
        output = self.linear3(output)
        return output

def get_rally_data(rally_idx, dataset):
    """Helper function for getting usable data from a dataset"""

    rally = dataset[rally_idx]
    states = torch.from_numpy(rally['states']).float().to(device)
    actions = torch.from_numpy(rally['actions']).float().to(device)
    rewards = torch.from_numpy(rally['rewards']).float().to(device)

    # Form s,a pairs
    sa_pairs = torch.cat([states, actions], -1)

    return sa_pairs, rewards

def get_prediction_accuracy(pred):
    """Get the accuracy of the predication
    
    Parameters:
    - pred:np.ndarray
        - Expected to be a 2D array, where pred[0] holds predictions for winner's sa_pairs and pred[1] holds predictions for loser's sa_pairs
    """
    winner_win_probs = pred[0]
    loser_win_probs = pred[1]
    
    correct = (len(winner_win_probs[winner_win_probs >= 0.5]) 
                + len(loser_win_probs[loser_win_probs < 0.5]))
    total_preds = (winner_win_probs.shape[0] + loser_win_probs.shape[0])
    return correct / total_preds

# Train
def train(save=True, use_mc=False, init_state_dict:Dict[str,Any]=None, num_epochs=num_epochs) -> Tuple[TennisEvalNN, Report]:

    # Create models
    model_net, target_net = TennisEvalNN().to(device), TennisEvalNN().to(device)

    # Use init_state_dict if present
    if init_state_dict is not None:
        print("Init state dictionary was passed. Using that to initialize model_net.")
        model_net.load_state_dict(init_state_dict)

    # Initialize weights of target net with model net
    target_net.load_state_dict(model_net.state_dict())
    target_net.eval()

    # Optimizer and criterions
    optimizer = optim.Adam(model_net.parameters(), lr=learning_rate)
    MSELoss = nn.MSELoss().to(device)

    # For calculating ave.
    losses = []
    accuracies = []

    # Train for num_epochs
    log = Report(num_epochs)
    for e in range(num_epochs):

        # Mark for logging that this is the start of a new epoch
        pos = e

        # Training
        # Randomize order of which rallies to use
        model_net.train()
        dataset_len = len(train_dataset)
        rally_indices = [i for i in range(dataset_len)]
        random.shuffle(rally_indices)
        for rally_idx in rally_indices:

            # Get rally data
            sa_pairs, rewards = get_rally_data(rally_idx, train_dataset)

            # Evaluate Q(s,a)
            eval_q = model_net(sa_pairs)

            # Evaluate Q(s',a')
            target_q = torch.zeros(eval_q.shape, device=device)
            # If rally continued past the serve
            if (eval_q.shape[1] > 1):
                # Leave the Q(s',a') of the last element 0 because thats the terminal state
                target_q[:,0:-1] = target_net(sa_pairs[:,1:]).detach()

            # TD target
            if not use_mc:
                y = rewards + gamma*target_q
            # MC
            else:
                for i in range(len(rewards[0])):
                    rewards[0, i, 0] = 1
                y = rewards

            loss = MSELoss(eval_q, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            accuracy = get_prediction_accuracy(eval_q)
            losses.append(loss.item())
            accuracies.append(accuracy)
            
            # Log and track for graphs
            pos += (1 / dataset_len)
            log.record(pos, train_loss=loss.item(), train_acc=accuracy, end='\r')

        # Print ave's
        ave_loss = np.mean(losses)
        ave_acc = np.mean(accuracies)

        # Log and track for graphs
        log.record(e + 1, train_ave_loss=ave_loss, train_ave_acc=ave_acc, end='\r')
        print(f" ave_loss: {ave_loss} ave_acc: {ave_acc}")
        
        # Reset average trackers
        losses = []
        accuracies = []


        # Sync target net with model net at certain intervals
        if e % 5 == 0:
            target_net.load_state_dict(model_net.state_dict())
    
    if save:
        os.makedirs(model_dirpath)
        torch.save(model_net.state_dict(), model_filepath)

    # eval()

    return model_net, log
        
# Evaluate
def eval(model):
    
    # Retsore model and set to eval mode
    # model = TennisEvalNN().to(device)
    # model.load_state_dict(torch.load(model_filepath))
    model.eval()

    correct, false = 0, 0
    mse = 0
    num = 0
    for rally_idx, rally in enumerate(test_dataset):
        rally = test_dataset[rally_idx]
        states = torch.from_numpy(rally['states']).float().to(device)
        actions = torch.from_numpy(rally['actions']).float().to(device)
        rewards = torch.from_numpy(rally['rewards']).float().to(device)
        sa_pairs = torch.cat([states, actions], -1)
        
        win_probs = model(sa_pairs).cpu().detach().numpy()

        winner_win_probs = win_probs[0].reshape(-1)
        loser_win_probs = win_probs[1].reshape(-1)

        num += 2 * len(winner_win_probs)

        correct += (len(winner_win_probs[winner_win_probs >= 0.5]) + len(loser_win_probs[loser_win_probs < 0.5]))
        false += (len(winner_win_probs[winner_win_probs < 0.5]) + len(loser_win_probs[loser_win_probs >= 0.5]))

        mse += sum([(1 - win_prob)**2 for win_prob in winner_win_probs])
        mse += sum([(0 - win_prob)**2 for win_prob in loser_win_probs])

        print('Accuracy:', correct/(correct+false))
        print('MSE:', mse / num)

    print(f'Total num: {(correct + false)}')

    print('Accuracy:', correct / (correct + false))
    print('MSE:', mse / num)

if __name__ == 'main':
    train()
    # eval()