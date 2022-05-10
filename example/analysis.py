import numpy as np
import torch
from NimEnvironments import NimEnv
from monte_carlo_tree_search import MCTS
from model import Nim_Model
import string
from main import set_seed
set_seed(30)

# board size: [1, 3, 5, 7, 9]
initial_pos = [2]
initial_pos.extend([1 for _ in range(1, 15)])
# test_position = [2, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1]
test_position = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
num_simulation = 100
print(test_position)

# initial_pos = [1, 3, 5, 7, 9]
# test_position = [1, 3, 2, 7, 2]
# num_simulation = 100
# print(test_position)


game = NimEnv(initial_pos=initial_pos)
game.reset()
state = game.position_to_state(test_position)

model = Nim_Model(action_size=game.action_size,
                  hidden_size=128,
                  num_lstm_layers=1,
                  num_head_layers=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(f'./models/15_200', map_location=device))

args = {'num_simulations': num_simulation}
mcts = MCTS(game, model, args)
root = mcts.run(state, game.to_play(), is_train=False)

total_visit_counts = np.array([child.visit_count for child in root.children.values()]).sum()

heap_indices = list(string.ascii_lowercase)
root_childrens = root.children.items()
root_childrens = sorted(root_childrens, key=lambda child: child[1].prior, reverse=True)

print('All the statistics are draw from the perspective of the player who is taking the move')
for i, (action, node) in enumerate(root_childrens):
    child_state = []
    sum_counter = 0
    
    # if the node has been visited
    if node.state is not None: 
        child_state = game.state_to_position(node.state)
        for idx, child_heap in enumerate(zip(child_state, test_position)):
            if child_heap[0] != child_heap[1]:
                removed_matches = child_heap[1] - child_heap[0]
                child_move = f'{heap_indices[idx]}{removed_matches}'

        print(f'{child_move}: {child_state} {not game.is_winning_position(game.state_to_position(node.state))}', end='   ')
        print(f'P:{node.prior}', end="  ")
        _, value = model.predict(node.state)
        print(f'V:{-value}({(0.5-value/2)*100}%)', end=" ")
        print(f'N: {node.visit_count}({(node.visit_count/total_visit_counts)*100}%)', end='   ')
        print(f'Q value:{-node.value()}', end='   ')
        print(f'WL:{0.5-node.value()/2}')

_, value = model.predict(root.state)
print(f'root:{test_position} {not game.is_winning_position(test_position)} ', end='')
print(f'V:{value}', end=" ")
print(f'WL:{(0.5 + root.value()/2)*100}%')
print(num_simulation)


