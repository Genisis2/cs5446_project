import numpy as np
import torch
from typing import Callable, List, Dict, Union

def score_model(test_dataset:List[Dict[str, np.ndarray]], get_win_prediction:Callable[[np.ndarray], Union[np.ndarray, torch.Tensor]]):
    """Scores the performance of a model based on the given dataset
    
    Parameters:
    - test_dataset:List[Dict[str, np.ndarray]]
        - The dataset to use to score the model

    - get_win_prediction:function
        - Function that receives a $(s,a)$ pair for the model to calculate $Q(s,a)$ for. Return of the
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
        pred_win_win_probs.append(pred)

    # Get loser predicted winning probabilities
    pred_loser_win_probs = []
    for sa_pair in lose_sa_pairs:
        pred = get_win_prediction(sa_pair)
        pred_loser_win_probs.append(pred)

    # Count the number of correct predictions
    correct_pred = 0
    # Also keep track of the squared errors
    squared_errors = []
    for pred in pred_win_win_probs:
        pred = pred.item()
        squared_errors.append((1 - pred)**2)
        if pred >= 0.5:
            correct_pred += 1
    for pred in pred_loser_win_probs:
        pred = pred.item()
        squared_errors.append((0 - pred)**2)
        if pred < 0.5:
            correct_pred += 1

    # Final score
    score = correct_pred / (len(pred_win_win_probs) + len(pred_loser_win_probs))
    # MSE
    mse = np.mean(squared_errors)

    return score, mse

def score_model_sequential(test_dataset:List[Dict[str, np.ndarray]], get_win_prediction:Callable[[np.ndarray], Union[np.ndarray, torch.Tensor]]):
    """Scores the performance of a model based on the given dataset. Sequential data is passed.
    
    Parameters:
    - test_dataset:List[Dict[str, np.ndarray]]
        - The dataset to use to score the model

    - get_win_prediction:function
        - Function that receives a $(s,a)$ pair for the model to calculate $Q(s,a)$ for. Return of the
        function should be $Q(s,a)$
    """

    # Get sa_pairs for winners and losers
    win_sa_pairs_seq = []
    lose_sa_pairs_seq = []
    for rally in test_dataset:
        for p_idx, (s, a) in enumerate(zip(rally['states'], rally['actions'])):
            # Get sa_pair
            sa_pairs = np.concatenate((s, a), axis=-1)
            # Put in correct list
            if (p_idx == 0):
                win_sa_pairs_seq.append(sa_pairs)
            else:
                lose_sa_pairs_seq.append(sa_pairs)
            
    # Get winner predicted winning probabilities
    pred_win_win_probs = []
    for sa_pair_seq in win_sa_pairs_seq:
        pred = get_win_prediction(sa_pair_seq)
        pred_win_win_probs.extend(pred)

    # Get loser predicted winning probabilities
    pred_loser_win_probs = []
    for sa_pair_seq in lose_sa_pairs_seq:
        pred = get_win_prediction(sa_pair_seq)
        pred_loser_win_probs.extend(pred)

    # Count the number of correct predictions
    correct_pred = 0
    # Also keep track of the squared errors
    squared_errors = []
    for pred in pred_win_win_probs:
        pred = pred.item()
        squared_errors.append((1 - pred)**2)
        if pred >= 0.5:
            correct_pred += 1
    for pred in pred_loser_win_probs:
        pred = pred.item()
        squared_errors.append((0 - pred)**2)
        if pred < 0.5:
            correct_pred += 1

    # Final score
    score = correct_pred / (len(pred_win_win_probs) + len(pred_loser_win_probs))
    # MSE
    mse = np.mean(squared_errors)

    return score, mse

def get_ave_stroke_q_per_rally(test_dataset:List[Dict[str, np.ndarray]], 
                        get_win_prediction:Callable[[np.ndarray], Union[np.ndarray, torch.Tensor]],
                        use_seq:bool=False):
    """Gets the ave. est. Q-value for each s,a pair per rally in the given dataset
    
    Parameters:
    - test_dataset:List[Dict[str, np.ndarray]]
        - The dataset to use to score the model

    - get_win_prediction:function
        - Function that receives a $(s,a)$ pair for the model to calculate $Q(s,a)$ for. Return of the
        function should be $Q(s,a)$
    
    - use_seq:bool
        - True if entire rally sequence will be passed. Otherwise, strokes will be passed one by one. Default false.
    """

    # Get sa_pairs for winners and losers
    win_sa_pairs_seq = []
    lose_sa_pairs_seq = []
    for rally in test_dataset:
        for p_idx, (s, a) in enumerate(zip(rally['states'], rally['actions'])):
            # Get sa_pair
            sa_pairs = np.concatenate((s, a), axis=-1)
            # Put in correct list
            if (p_idx == 0):
                win_sa_pairs_seq.append(sa_pairs)
            else:
                lose_sa_pairs_seq.append(sa_pairs)
            
    # Get winner predicted winning probabilities
    pred_win_win_probs = []
    for sa_pair_seq in win_sa_pairs_seq:
        
        # Will hold the Q for each stroke
        stroke_win_preds = []

        # Get the predictions for each stroke
        if use_seq:
            pred = get_win_prediction(sa_pair_seq)
            stroke_win_preds.extend(pred)
        else:
            for sa_pair in sa_pair_seq:
                pred = get_win_prediction(sa_pair)
                stroke_win_preds.append(pred)
        
        # Store in the overall pred storage
        for stroke_idx, stroke_pred in enumerate(stroke_win_preds):
            # Create list for pred for stroke if not yet existing
            if stroke_idx + 1 > len(pred_win_win_probs):
                pred_win_win_probs.append([])
            # Store pred in appropriate stroke list
            stroke_pred = stroke_pred.item()
            pred_win_win_probs[stroke_idx].append(stroke_pred)

    # Get loser predicted winning probabilities
    pred_loser_win_probs = []
    for sa_pair_seq in lose_sa_pairs_seq:
        
        # Will hold the Q for each stroke
        stroke_win_preds = []

        # Get the predictions for each stroke
        if use_seq:
            pred = get_win_prediction(sa_pair_seq)
            stroke_win_preds.extend(pred)
        else:
            for sa_pair in sa_pair_seq:
                pred = get_win_prediction(sa_pair)
                stroke_win_preds.append(pred)
        
        # Store in the overall pred storage
        for stroke_idx, stroke_pred in enumerate(stroke_win_preds):
            # Create list for pred for stroke if not yet existing
            if stroke_idx + 1 > len(pred_loser_win_probs):
                pred_loser_win_probs.append([])
            # Store pred in appropriate stroke list
            stroke_pred = stroke_pred.item()
            pred_loser_win_probs[stroke_idx].append(stroke_pred)

    # Calculate the average prediction per stroke
    winner_ave_pred_per_stroke = []
    for stroke_preds in pred_win_win_probs:
        winner_ave_pred_per_stroke.append(np.mean(stroke_preds))
    loser_ave_pred_per_stroke = []
    for stroke_preds in pred_loser_win_probs:
        loser_ave_pred_per_stroke.append(np.mean(stroke_preds))

    return winner_ave_pred_per_stroke, loser_ave_pred_per_stroke
