import numpy as np


class NimEnv:
    def __init__(self, initial_pos=[2, 1, 1, 1, 1]):
        super(NimEnv, self).__init__()
        self.nim_game = NimUnitary(initial_pos)
        self.nim_game_copy = NimUnitary(initial_pos)

        self.action_size = self.nim_game.action_size
        self.player = 1
    
    def to_play(self):
        return self.player

    def get_next_state(self, state, action):
        self.nim_game_copy.board = state.copy()
        self.nim_game_copy.action(action)

        next_state = self.nim_game_copy.observe()
        done = self.nim_game_copy.is_done()
        reward = 1.0 if done else 0.0

        return next_state, reward, done

    def reset(self):
        self.player = 1
        self.nim_game.initialize()
        return self.nim_game.observe()
    
    def step(self, action):
        self.nim_game.action(action)
        
        done = self.nim_game.is_done()
        reward = 1.0 if done else 0.0
        obs = self.nim_game.observe()
        
        self.player *= -1
        
        return obs.copy(), reward, done

    def get_action_mask(self, state):
        return self.nim_game.get_action_mask(state)
       

class NimUnitary(object):
    def __init__(self, initial_pos=[2, 1, 1, 1, 1]):
        self.initial_pos = initial_pos
        self.num_heaps = len(initial_pos)
        self.action_size = sum(initial_pos)
        self.action_space = []
        self.board_size = np.sum(initial_pos, dtype=np.int32) + self.num_heaps - 1
        self.position = initial_pos
        self.board = np.ones((self.board_size,), dtype=np.float64)
        self.sep_idx = [-1]
    
    def initialize(self):
        idx = -1
        for num_heaps in self.initial_pos[:-1]:
            idx += (num_heaps + 1)
            self.board[idx] = -1
        self.sep_idx.extend(np.where(self.board == -1)[0])

        for idx, n_heaps in enumerate(self.initial_pos):
            for n in range(1, n_heaps + 1):
                self.action_space.append((idx, n))

    def action(self, action):  # (0, 1)
        heap_idx, n_counters = action
        heap_idx = self.sep_idx[heap_idx]
        idx = 1
        while n_counters > 0:
            if self.board[heap_idx + idx] == 0.0:
                idx += 1
            elif self.board[heap_idx + idx] == 1.0:
                self.board[heap_idx + idx] = 0.0
                n_counters -= 1

    def get_action_mask(self, position):
        mask = np.zeros((self.action_size,), dtype=np.float64)
        n_heaps = 0
        n_counter = 0
        for counter in position:
            if counter == 1:
                n_counter += 1
                action_idx = self.action_space.index((n_heaps, n_counter))
                mask[action_idx] = 1.0
            elif counter == -1:
                n_heaps += 1
                n_counter = 0
        return mask
    
    def is_done(self):
        if self.board.sum() == (self.num_piles - 1) * -1.0:
            return True
        else:
            return False
    
    def observe(self):
        return self.board.copy()


if __name__ == '__main__':
    nim = NimUnitary(initial_pos=[1, 3, 5, 7])
    nim.initialize()
    print(nim.action_space)
    print((0, 1))
    nim.action((0, 1))
    print(nim.board)
    print(nim.get_action_mask())
    print((2, 5))
    nim.action((2, 5))
    print(nim.board)
    print(nim.get_action_mask())
    print((3, 3))
    nim.action((3, 3))
    print(nim.board)
    print(nim.get_action_mask())