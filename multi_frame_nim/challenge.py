from game import Nim
from mcts import MCTS
from model import Nim_Model
import torch
from collections import Counter

def compete(player_1, n_round=100):
    results = []
    
    for _ in range(n_round):
        state = game.reset()
        done = False
        while not done:
            root = player_1.run(state, game.to_play(), is_train=False)
            action = root.select_action(temperature=0.0)
            _, reward, done = game.step(action)
            if done:
                results.append(reward) 
                break
            
            state, reward, done = game.step(game.winning_move())
            # print(next_state, game.unpack_action(action))
            if done: 
                results.append(-reward)
                break

    return results

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

board = [2 for _ in range(10)]
board[9] = 1

for include_history in [True, False]:
    if include_history:
        print("With history", end="    ")
    else:
        print("Without history", end=" ")

    game = Nim(init_board=board, 
            include_history=include_history)

    model = Nim_Model(action_size=game.action_size, hidden_size=16, num_lstm_layers=1, num_head_layers=1).to(device)
    model.load_state_dict(torch.load(f'./models/{game.n_piles}_{game.include_history}_0.35/final', weights_only=True, map_location=device))

    args = {'num_simulations': 20,  
            'alpha': 0.35}
    player = MCTS(game, model, args)

    results = compete(player, 100)
    print(Counter(results))