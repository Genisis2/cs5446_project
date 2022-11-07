import numpy as np
import torch
from typing import Callable, List, Dict, Union

def score_model(test_dataset:List[Dict[str, np.ndarray]], get_win_prediction:Callable[[np.ndarray], Union[np.ndarray, torch.Tensor]]):
    """Scores the performance of a model based on the given dataset
    
    Parameters:
    - test_dataset:List[Dict[str, np.ndarray]]

        The dataset to use to score the model

    - get_win_prediction:function
        
        Function that receives a $(s,a)$ pair for the model to calculate $Q(s,a)$ for. Return of the
        function should be $Q(s,a)$
    """

    # Get sa_pairs for winners and losers
    win_sa_pairs = []
    lose_sa_pairs = []
    for rally in test_dataset:
        for p_idx, (s, a) in enumerate(zip(rally['states'], rally['actions'])):
            # Get sa_pair
            sa_pairs = np.concatenate((s, a), axis=-1)
            # Put in correct list
            if (p_idx == 0):
                win_sa_pairs.extend(sa_pairs)
            else:
                lose_sa_pairs.extend(sa_pairs)
            
    # Get winner predicted winning probabilities
    pred_win_win_probs = []
    for sa_pair in win_sa_pairs:
        pred = get_win_prediction(sa_pair)
        pred_win_win_probs.append(pred.item())

    # Get loser predicted losing probabilities
    pred_loser_lose_probs = []
    for sa_pair in lose_sa_pairs:
        pred = get_win_prediction(sa_pair)
        pred_loser_lose_probs.append(pred.item())

    # Count the number of correct predictions
    correct_pred = 0
    for pred in pred_win_win_probs:
        if pred >= 0.5:
            correct_pred += 1
    for pred in pred_loser_lose_probs:
        if pred < 0.5:
            correct_pred += 1

    # Final score
    score = correct_pred / (len(pred_win_win_probs) + len(pred_loser_lose_probs))

    print(f"Accuracy: {score}")

    return score
