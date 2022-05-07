from NimEnvironments import NimEnv
import numpy as np


def winning_policy_value(state, num_piles=6):
    game = NimEnv(num_piles=num_piles)
    game.set_board(state)
    mask = game.get_action_masks()
    
    policies = []
    legal_actions = game.get_all_legal_actions() 
    actions = []

    for action in legal_actions:
        game.set_board(state)
        
        game.nim_unitary.action(action)
        board = game.nim_unitary.board
        
        result = 0
        for pile_index in range(game.nim_unitary.num_piles):
            start_pile_index = pile_index**2 + pile_index
            end_pile_index = (pile_index + 1)**2 + pile_index
            pile = board[start_pile_index: end_pile_index]
            matches = sum(pile)
            result = np.bitwise_xor(result, int(matches))
        if result == 0:
            winning_move = [0.0 for _ in range(len(game.all_actions))]
            
            idx = game.all_actions.index(action)
            winning_move[idx] = 1.0
            policies.append(winning_move)
            actions.append(action)

    if len(policies) > 0:
        value = 1
    else:
        value = -1
    return policies, value, mask
    

def get_state_space(num_pile=6):
    state_list = []
    for i in range(num_pile):
        num_matches = 2 * i + 1
        states = []
        state = [0.0 for _ in range(num_matches)]
        
        states.append(state.copy())
        for j in range(1, num_matches+1):
            state[-j] = 1.0
            states.append(state.copy())
            
        state_list.append(states.copy())

    state_space = []
    if num_pile == 7:
        a, b, c, d, e, f, g = state_list
        for ai in a:
            for bi in b:
                for ci in c:
                    for di in d:
                        for ei in e:
                            for fi in f:
                                for gi in g:
                                    state = ai + [-1.0] + bi + [-1.0] + ci + [-1.0] + di + [-1.0] + ei + [-1.0] + fi + [-1.0] + gi
                                    state_space.append(state.copy())
    elif num_pile == 6:
        a, b, c, d, e, f = state_list
        for ai in a:
            for bi in b:
                for ci in c:
                    for di in d:
                        for ei in e:
                            for fi in f:
                                state = ai + [-1.0] + bi + [-1.0] + ci + [-1.0] + di + [-1.0] + ei + [-1.0] + fi
                                state_space.append(state.copy())
    elif num_pile == 5:
        a, b, c, d, e = state_list
        for ai in a:
            for bi in b:
                for ci in c:
                    for di in d:
                        for ei in e:
                            state = ai + [-1.0] + bi + [-1.0] + ci + [-1.0] + di + [-1.0] + ei
                            state_space.append(state.copy())
    elif num_pile == 4:
        a, b, c, d = state_list
        for ai in a:
            for bi in b:
                for ci in c:
                    for di in d:
                        state = ai + [-1.0] + bi + [-1.0] + ci + [-1.0] + di
                        state_space.append(state.copy())
    elif num_pile == 3:
        a, b, c = state_list
        for ai in a:
            for bi in b:
                for ci in c:
                    state = ai + [-1.0] + bi + [-1.0] + ci
                    state_space.append(state.copy())
    return state_space


def get_states_policies_values_masks(num_pile=6, num_samples=10000):
    state_space = get_state_space(num_pile=num_pile)
    policies = []
    values = []
    masks = []
    
    for state in state_space:
        policy, value, mask = winning_policy_value(state, num_piles=num_pile)

        policies.append(policy.copy())
        values.append(value)
        masks.append(mask)

    idx = np.random.permutation(len(state_space))
    state_space = np.array(state_space)[idx][:num_samples]
    policies = np.array(policies, dtype=object)[idx][:num_samples]
    values = np.array(values)[idx][:num_samples]
    masks = np.array(masks)[idx][:num_samples]

    return state_space.tolist(), policies.tolist(), values.tolist(), masks.tolist()


if __name__ == '__main__':
   states, policies, values, masks = get_states_policies_values_masks(num_pile=4)
   state_space = get_state_space(num_pile=3)
   print(states)
   print(masks)

