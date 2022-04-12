import numpy as np


class NimEnv:
    def __init__(self, num_piles=3):
        super(NimEnv, self).__init__()
        self.num_piles = num_piles
        self.nim_unitary = NimUnitary(num_piles)
        self.all_actions = self.get_all_legal_actions()
        self.board_size = self.nim_unitary.board_size
        self.action_size = len(self.all_actions)
        self.player = 1

    def set_board(self, state):
        state = state.copy()
        self.nim_unitary.board = state
    
    def get_observation(self):
        observation = self.nim_unitary.observe()
        return observation.copy()
    
    def to_play(self):
        return self.player

    def reset(self):
        self.player = 1
        self.nim_unitary._initialize_board()
        return self.get_observation()
    
    def step(self, action):
        action = self.all_actions[action]
        self.nim_unitary.action(action)
        
        done = self.nim_unitary.is_done()
        reward = 1.0 if done else 0.0
        obs = self.get_observation()
        
        self.player *= -1
        
        return obs.copy(), reward, done

    def sample_random_action(self):
        mask = np.array(self.get_action_masks())
        action = np.random.choice(np.where(mask == 1)[0])
        return action

    def get_action_masks(self):
        mask = [0.0 for _ in range(len(self.all_actions))]
        all_legal_action_indices = self.get_all_legal_action_indices()
        for idx in all_legal_action_indices:
            mask[idx] = 1.0
        return mask.copy()
    
    def get_all_legal_actions(self):
        legal_actions = []
        for pile_index in range(self.nim_unitary.num_piles):
            start_pile_index = pile_index ** 2 + pile_index
            end_pile_index = (pile_index + 1) ** 2 + pile_index

            pile = self.nim_unitary.board[start_pile_index: end_pile_index]
            num_matches = int(sum(pile))
            if num_matches > 0:
                for match in range(num_matches):
                    legal_actions.append((pile_index, match+1))
        return legal_actions.copy()

    def get_all_legal_action_indices(self):
        legal_action_indices = []
        for action in self.get_all_legal_actions():
            index = self.all_actions.index(action)
            legal_action_indices.append(index)
        return legal_action_indices.copy()
            
    def decimal_repr(self):
        decimal_state = []
        for pile_index in range(self.nim_unitary.num_piles):
            start_pile_index = pile_index ** 2 + pile_index
            end_pile_index = (pile_index + 1) ** 2 + pile_index

            pile = self.nim_unitary.board[start_pile_index: end_pile_index]
            num_matches = int(sum(pile))
            decimal_state.append(num_matches)
        return decimal_state


class NimUnitary(object):
    def __init__(self, num_piles=3):
        self.num_piles = num_piles
        self.num_matches = num_piles ** 2 
        self.board_size = (self.num_piles - 1) + self.num_matches
        self._initialize_board()
        self.rewards = {'WON': 1.0,
                        'VALID_ACTION': 0}
        
    def _initialize_board(self):
        self.is_action_valid = True
        self.board = np.ones((self.board_size,), dtype=np.float64)
        for i in range(1, self.num_piles):
            self.board[i**2 + i-1] = -1  # noise used to separate piles
    
    def action(self, action):
        pile_index, match_num = action
        start_pile_index = pile_index**2 + pile_index
        end_pile_index = (pile_index + 1)**2 + pile_index
        pile = self.board[start_pile_index: end_pile_index]
        count = 1
        for i in range(len(pile)):
            if (pile[i] == 1.0) and (count <= match_num):
                pile[i] = 0.0
                count += 1
        self.board[start_pile_index: end_pile_index] = pile
        
    def evaluate(self):
        if self.board.sum() == (self.num_piles - 1) * -1.0:
            return self.rewards['WON']
        else:
            return self.rewards['VALID_ACTION']
    
    def is_done(self):
        if self.board.sum() == (self.num_piles - 1) * -1.0:
            return True
        else:
            return False
    
    def observe(self):
        return self.board.copy()
