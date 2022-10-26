import numpy as np
import torch
from sklearn.model_selection import train_test_split

from data import train_dataset, test_dataset


def train() -> None:
    print(len(train_dataset), len(test_dataset))


def test() -> None:
    pass


if __name__ == '__main__':
    train()
