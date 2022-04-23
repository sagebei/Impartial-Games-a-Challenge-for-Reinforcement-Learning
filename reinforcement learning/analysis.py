import numpy as np
import torch
from NimEnvironments import NimEnv
from monte_carlo_tree_search import MCTS
from model import Nim_Model
from main import set_seed
set_seed(30)

# board size: [1, 3, 5, 7, 9]
heaps = [1, 3, 3, 0, 0]
num_simulation = 10000

state = []
for i, counters in enumerate(heaps):
    num_counters = 2 * i + 1
    heap = [0 for _ in range(num_counters)]
    for c in range(1, counters+1):
        heap[-c] = 1
    if i < len(heaps) - 1:
        heap.append(-1)
    state.extend(heap)

# print(heaps)
# print(state)
state = np.array(state, dtype=np.float64)

game = NimEnv(num_piles=len(heaps))
model = Nim_Model(action_size=game.action_size,
                  hidden_size=128,
                  num_layers=1)
model.load_state_dict(torch.load('./models/5_1'))
args = {'num_simulations': num_simulation,  
        'alpha': 0.35,
        'c_puct': 3}
mcts = MCTS(game, model, args)
root = mcts.run(state, game.to_play(), is_train=False)

visit_counts = np.array([child.visit_count for child in root.children.values()])
visit_count_distribution = visit_counts / sum(visit_counts)

for i, (action, node) in enumerate(root.children.items()):
    child_state = []
    sum_counter = 0
    for counter in list(np.array(node.state, dtype=np.int8)):
        if counter == -1:
            child_state.append(sum_counter)
            sum_counter = 0
        else:
            sum_counter += counter
    child_state.append(sum_counter)
    
    print(f'Child: {child_state}', end='   ')
    print(f'P:{node.prior}', end="  ")
    _, value = model.predict(node.state)
    print(f'V:{value}', end=" ")
    print(f'N: {node.visit_count}({visit_count_distribution[i]}%)', end='   ')
    print(f'Q value:{node.value()}')

_, value = model.predict(root.state)
print(f'root:{heaps} ', end='')
print(f'V:{value}', end=" ")
print(f'Q value:{root.value()}')

