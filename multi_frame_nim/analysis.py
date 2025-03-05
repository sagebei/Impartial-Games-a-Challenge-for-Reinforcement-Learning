import numpy as np
import torch
from game import Nim
from mcts import MCTS
from model import Nim_Model
from main import set_seed
set_seed(30)


board = [2 for _ in range(10)]
board[0] = 0
board[1] = 0
# board[2] = 0
# board[3] = 0
board[4] = 1
history = [2, 1, -1]
num_simulation = 100
include_history = False

if include_history:
    print("include history")
else:
    print("not include history")

def win_lose_position(board):
    xor = 0
    for c in board:
        xor = c ^ xor
    if xor == 0:
        win_lost = 'WIN '
    else: 
        win_lost = 'LOSE'
    return win_lost

game = Nim(board, 
           include_history)
model = Nim_Model(action_size=game.action_size,
                  hidden_size=16,
                  num_lstm_layers=1,
                  num_head_layers=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(f'./models/{len(board)}_{game.include_history}_0.35/3700', weights_only=True, map_location=device))


mcts = MCTS(game, model, {'num_simulations': 
                          num_simulation})
if game.include_history:
    root = mcts.run(history + board, game.to_play(), is_train=False)
else:
    root = mcts.run(board, game.to_play(), is_train=False)

_, value = model.predict(root.state)
print(f'root:{board} {win_lose_position(board)} ', end='')
print(f'V:{round(value)}', end=" ")
print(f'WL:{round((0.5 + root.value()/2)*100, 2)}%')

total_visit_counts = np.array([child.visit_count for child in root.children.values()]).sum()

root_childrens = root.children.items()
root_childrens = sorted(root_childrens, key=lambda child: child[1].visit_count, reverse=True)
# print('From the perspective of the player who act on the state of the root node')
for i, (action, child) in enumerate(root_childrens):
    # if the node has been visited
    if child.state is not None: 
        if game.include_history:
            child_board = child.state[3:]
        else: 
            child_board = child.state
        print(f'{game.unpack_action(action)}: {child_board} {win_lose_position(child_board)}', end='   ')
        print(f'P:{round(child.prior, 2)}', end="  ")
        _, value = model.predict(child.state)
        print(f'V:{round(-value, 2)}({round((0.5-value/2)*100, 2)}%)', end=" ")
        print(f'N: {round(child.visit_count, 2)}({(child.visit_count/total_visit_counts)*100}%)', end='   ')
        print(f'Q value:{round(-child.value(), 2)}', end='   ')
        print(f'WL:{round(0.5-child.value()/2, 2)}')