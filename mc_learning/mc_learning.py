import numpy as np
import torch
from sklearn.model_selection import train_test_split

from data import trails 

def train():
    print(trails)

def test():
    pass

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = train_test_split(trails)