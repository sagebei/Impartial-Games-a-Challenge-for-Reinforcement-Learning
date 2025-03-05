import numpy as np


class Nim:
    def __init__(self, init_board=[2, 2, 2, 2, 2], include_history=True):
        super(Nim, self).__init__()
        self.init_board = init_board
        self.n_piles = len(init_board)
        self.board = [2 * i + 1 for i in range(self.n_piles)]
        self.history = []

        self.all_legal_actions = self.legal_actions()
        self.all_legal_actions_idx = {action: idx for idx, action in enumerate(self.all_legal_actions)}
        self.action_size = len(self.all_legal_actions)

        self.player = 1
        self.include_history = include_history

    def reset_board(self, state):
        state = state.copy()
        if self.include_history:
            self.history = state[0:3]
            self.board = state[3:]
        else:
            self.board = state
    
    def to_play(self):
        return self.player

    def reset(self):
        self.board = self.init_board.copy()
        if self.include_history:
            self.history = [0, 0, -1]
        return self.state()
    
    def state(self):
        if self.include_history:
            state = self.history + self.board
        else:
            state = self.board

        return state.copy()
    
    def step(self, action):
        pile_idx, take = self.unpack_action(action)
        before_take = self.board[pile_idx]
        after_take = before_take - take

        if self.include_history:
            self.history = [before_take, after_take, -1]
        self.board[pile_idx] = after_take
        
        done = sum(self.board) == 0
        reward = 1.0 if done else 0.0
        state = self.state()
        
        self.player *= -1
        
        return state.copy(), reward, done
    
    def legal_actions(self):
        actions = []
        for (pile_idx, take) in enumerate(self.board):
            if take > 0:
                for i in range(take):
                    action = i * self.n_piles + pile_idx
                    actions.append(action)
        return actions.copy()
    
    def action_masks(self):
        mask = [0.0 for _ in range(len(self.all_legal_actions))]
        legal_actions = self.legal_actions()
        for i, action in enumerate(self.all_legal_actions):
            if action in legal_actions:
                mask[i] = 1.0
        return mask.copy()
        
    def unpack_action(self, action):
        pile_idx = action % self.n_piles
        take = int((action - pile_idx) / self.n_piles + 1)
        return pile_idx, take
    

    def winning_position(self, board):
        xor = 0
        for c in board:
            xor = c ^ xor
        if xor == 0:
            return True
        else: 
            return False
        
    def winning_move(self):
        actions = self.legal_actions()
        for action in actions:
            board = self.board.copy()
            pile_idx, take = self.unpack_action(action)
            board[pile_idx] = self.board[pile_idx] - take
            if self.winning_position(board):
                return action
            
        return np.random.choice(actions)


if __name__ == "__main__":
    nim = Nim(init_board=[2, 2, 2], include_history=True)
    state = nim.reset()
    done = False
    print(state)
    while not done:
       action = nim.random_action()
       print(nim.unpack_action(action))
       state, reward, done = nim.step(action)
    #    print(nim.action_masks())
       print(state)
