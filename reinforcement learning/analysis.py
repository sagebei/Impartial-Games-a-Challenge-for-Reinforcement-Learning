import numpy as np
from NimEnvironments import NimEnv
from monte_carlo_tree_search import MCTS
from model import Nim_Model

# board size: [1, 3, 5, 7, 9]
heaps = [1, 3, 5, 0, 0]
num_simulation = 1000

state = []
for i, counters in enumerate(heaps):
    num_counters = 2 * i + 1
    heap = [0 for _ in range(num_counters)]
    for c in range(1, counters+1):
        heap[-c] = 1
    if i < len(heaps) - 1:
        heap.append(-1)
    state.extend(heap)

print(heaps)
print('root node:', end=' ')
print(state)
state = np.array(state, dtype=np.float64)

game = NimEnv(num_piles=len(heaps))
model = Nim_Model(action_size=game.action_size,
                  hidden_size=128,
                  num_layers=1)
args = {'num_simulations': num_simulation,  
        'alpha': 0.35,
        'c_puct': 3}
mcts = MCTS(game, model, args)
root = mcts.run(state, game.to_play(), is_train=False)

visit_counts = np.array([child.visit_count for child in root.children.values()])
visit_count_distribution = visit_counts / sum(visit_counts)

for i, (action, node) in enumerate(root.children.items()):
    
    print('P:', end='')
    print(node.prior, end="  V:")
    _, value = model.predict(node.state)
    print(value, end="  Child:")
    child_node = []
    sum_counter = 0
    for counter in list(np.array(node.state, dtype=np.int8)):
        if counter == -1:
            child_node.append(sum_counter)
            sum_counter = 0
        else:
            sum_counter += counter
    child_node.append(sum_counter)
    print(child_node, end='  MCTS Prob: ')
    print(visit_count_distribution[i])
