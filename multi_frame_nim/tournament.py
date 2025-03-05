from game import Nim
from mcts import MCTS
from model import Nim_Model
import torch
from collections import Counter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

board = [2 for _ in range(10)]
board[0] = 2
board[1] = 1
game_1 = Nim(init_board=board, 
             include_history=True)
game_2 = Nim(init_board=board, 
             include_history=False)

model_1 = Nim_Model(action_size=game_1.action_size, hidden_size=16, num_lstm_layers=1, num_head_layers=1).to(device)
model_1.load_state_dict(torch.load(f'./models/{game_1.n_piles}_{game_1.include_history}_0.35/3700', weights_only=True, map_location=device))

model_2 = Nim_Model(action_size=game_2.action_size, hidden_size=16, num_lstm_layers=1, num_head_layers=1).to(device)
model_2.load_state_dict(torch.load(f'./models/{game_2.n_piles}_{game_2.include_history}_0.35/3700', weights_only=True, map_location=device))

args = {'num_simulations': 10,  
        'alpha': 0.35}
player_1 = MCTS(game_1, model_1, args)
player_2 = MCTS(game_2, model_2, args)

def compete(player_1, player_2, n_round=100):
    results = []
    
    for _ in range(n_round // 2):
        state = game_1.reset()
        done = False
        while not done:
            root = player_1.run(state, game_1.to_play(), is_train=False)
            action = root.select_action(temperature=0.0)
            # print(state, game_1.unpack_action(action))
            next_state, reward, done = game_1.step(action)
            if done:
                results.append(reward) 
                break
            
            root = player_2.run(next_state[3:], game_1.to_play(), is_train=False)
            action = root.select_action(temperature=0.0)
            state, reward, done = game_1.step(action)
            # print(next_state, game.unpack_action(action))
            if done: 
                results.append(-reward)
                break

    for _ in range(n_round // 2):
        state = game_1.reset()
        done = False
        while not done:
            root = player_2.run(state[3:], game_1.to_play(), is_train=False)
            action = root.select_action(temperature=0.0)
            # print(state, game.unpack_action(action))
            next_state, reward, done = game_1.step(action)
            if done:
                results.append(-reward) 
                break
            
            root = player_1.run(next_state, game_1.to_play(), is_train=False)
            action = root.select_action(temperature=0.0)
            state, reward, done = game_1.step(action)
            # print(next_state, game.unpack_action(action))
            if done: 
                results.append(reward)
                break

    return results

results = compete(player_1, player_2, 100)
print(Counter(results))