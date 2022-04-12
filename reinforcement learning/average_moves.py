import numpy as np
from random import shuffle
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from NimEnvironments import NimEnv
from monte_carlo_tree_search import MCTS
from EloRating import Elo
from PlayerPool import PlayerPool
from ExpertPolicyValue import get_states_policies_values_masks
import ray

game = NimEnv(num_piles=5)

def execute_episode():
    _ = game.reset()
    done = False
    n_moves = 0
    while not done:
        action = game.sample_random_action()
        next_state, reward, done = game.step(action)

        n_moves += 1

        if done:
            return n_moves

moves = [execute_episode() for i in range(10000)]
print(sum(moves) / len(moves))