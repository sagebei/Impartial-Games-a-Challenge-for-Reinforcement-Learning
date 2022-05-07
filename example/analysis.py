import numpy as np
import torch
from NimEnvironments import NimEnv
from monte_carlo_tree_search import MCTS
from model import Nim_Model
from main import set_seed
set_seed(30)


# board size: [1, 3, 5, 7, 9]
heaps = [1, 3, 5, 5, 9]
num_simulation = 65536

def win_lose_position(position):
    xor = 0
    for c in child_state:
        xor = c ^ xor
    if xor == 0:
        win_lost = 'WIN'
    else: 
        win_lost = 'LOSE'
    return win_lost

state = []
for i, counters in enumerate(heaps):
    num_counters = 2 * i + 1
    heap = [0 for _ in range(num_counters)]
    for c in range(1, counters+1):
        heap[-c] = 1
    if i < len(heaps) - 1:
        heap.append(-1)
    state.extend(heap)

state = np.array(state, dtype=np.float64)

game = NimEnv(num_piles=len(heaps))
model = Nim_Model(action_size=game.action_size,
                  hidden_size=128,
                  num_lstm_layers=1,
                  num_head_layers=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(f'./models/{len(heaps)}heaps', map_location=device))

args = {'num_simulations': num_simulation,  
        'alpha': 0.35}
mcts = MCTS(game, model, args)
root = mcts.run(state, game.to_play(), is_train=False)

total_visit_counts = np.array([child.visit_count for child in root.children.values()]).sum()

heap_indices = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
root_childrens = root.children.items()
root_childrens = sorted(root_childrens, key=lambda child: child[1].prior, reverse=True)

print('All the statistics are draw from the perspective of the player who is taking the move')
for i, (action, node) in enumerate(root_childrens):
    child_state = []
    sum_counter = 0
    
    # if the node has been visited
    if node.state is not None: 
        for counter in list(np.array(node.state, dtype=np.int8)):
            if counter == -1:
                child_state.append(sum_counter)
                sum_counter = 0
            else:
                sum_counter += counter
        child_state.append(sum_counter)

        for idx, child_heap in enumerate(zip(child_state, heaps)):
            if child_heap[0] != child_heap[1]:
                removed_matches = child_heap[1] - child_heap[0]
                child_move = f'{heap_indices[idx]}{removed_matches}'

        print(f'{child_move}: {child_state} {win_lose_position(child_state)}', end='   ')
        print(f'P:{node.prior}', end="  ")
        _, value = model.predict(node.state)
        print(f'V:{-value}({(0.5-value/2)*100}%)', end=" ")
        print(f'N: {node.visit_count}({(node.visit_count/total_visit_counts)*100}%)', end='   ')
        print(f'Q value:{-node.value()}', end='   ')
        print(f'WL:{0.5-node.value()/2}')

_, value = model.predict(root.state)
print(f'root:{heaps} {win_lose_position(heaps)} ', end='')
print(f'V:{value}', end=" ")
print(f'WL:{(0.5 + root.value()/2)*100}%')
print(num_simulation)
