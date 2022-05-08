import numpy as np
import itertools
from copy import deepcopy


class NimEnv:
    def __init__(self, initial_pos=[2, 1, 1, 1, 1]):
        super(NimEnv, self).__init__()
        self.nim_game = NimUnitary(initial_pos)
        self.nim_game_copy = NimUnitary(initial_pos)
        self.action_size = self.nim_game.action_size
        self.action_space = self.nim_game.action_space
        self.player = 1
    
    def to_play(self):
        return self.player

    def get_next_state(self, state, action):
        if not isinstance(action, tuple):
            action = self.action_space[action]

        self.nim_game_copy.board = state.copy()
        self.nim_game_copy.action(action)
        next_state = self.nim_game_copy.observe()
        done = self.nim_game_copy.is_done()
        reward = 1.0 if done else 0.0

        return next_state, reward, done

    def reset(self):
        self.player = 1
        self.nim_game.initialize()
        self.nim_game_copy.initialize()
        return self.nim_game.observe()
    
    def step(self, action):
        if not isinstance(action, tuple):
            action = self.action_space[action]

        self.nim_game.action(action)
        done = self.nim_game.is_done()
        reward = 1.0 if done else 0.0
        obs = self.nim_game.observe()
        
        self.player *= -1
        
        return obs.copy(), reward, done

    def get_action_mask(self, state):
        return self.nim_game.get_action_mask(state)

    def convert_state(self, state):
        return self.nim_game.convert_state(state)
       

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

        self.initialize()
    
    def initialize(self):
        # mark the separator on the board
        idx = -1
        for num_counters in self.initial_pos[:-1]:
            idx += (num_counters + 1)
            self.board[idx] = -1
        self.sep_idx.extend(np.where(self.board == -1)[0])
        # construct the action space
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
    
    def is_done(self):
        if self.board.sum() == (self.num_heaps - 1) * -1.0:
            return True
        else:
            return False
    
    def observe(self):
        return self.board.copy()

    def position_to_state(self, position):
        # build empty state
        state = np.zeros((self.board_size,), dtype=np.float64)
        # mark the separator
        idx = -1
        for num_heaps in self.initial_pos[:-1]:
            idx += (num_heaps + 1)
            state[idx] = -1.0
        # add the counters
        sep_idx = np.where(self.board == -1.0)[0].tolist()
        sep_idx.append(-1)
        for idx, n_counters in zip(sep_idx, position):
            if idx == -1:
                for n in range(n_counters):
                    state[idx - n] = 1.0
            else:
                for n in range(1, n_counters + 1):
                    state[idx - n] = 1.0
        return state

    def state_to_position(self, state):
        position = []
        heap = 0
        for s in state:
            if s == -1:
                position.append(int(heap))
                heap = 0
            else:
                heap += s
        position.append(int(heap))
        return position

    def get_action_mask(self, state):
        mask = np.zeros((self.action_size,), dtype=np.float64)
        n_heaps = 0
        n_counter = 0
        for counter in state:
            if counter == 1:
                n_counter += 1
                action_idx = self.action_space.index((n_heaps, n_counter))
                mask[action_idx] = 1.0
            elif counter == -1:
                n_heaps += 1
                n_counter = 0
        return mask

    def get_state_space(self):
        heap = []
        for n in self.initial_pos:
            heap.append([i for i in range(n+1)])
        heaps = list(itertools.product(*heap))
        state_space = list(map(self.position_to_state, heaps))
        return state_space

    def get_states_policies_values_masks(self, num_samples=10000):
        state_space = self.get_state_space()
        policy_space = []
        value_space = []
        mask_space = []

        for state in state_space:
            position = self.state_to_position(state)
            policy, value = self.evaluate_position(position)
            mask = self.get_action_mask(state)
            policy_space.append(policy)
            value_space.append(value)
            mask_space.append(mask)

        idx = np.random.permutation(len(state_space))
        state_space = np.array(state_space)[idx][:num_samples]
        policy_space = np.array(policy_space, dtype=object)[idx][:num_samples]
        value_space = np.array(value_space)[idx][:num_samples]
        mask_space = np.array(mask_space)[idx][:num_samples]

        return state_space.tolist(), policy_space.tolist(), value_space.tolist(), mask_space.tolist()

    def move_on_position(self, position, move):
        position = deepcopy(position)
        heap_idx, n_counters = move
        position[heap_idx] -= n_counters
        return position

    def is_winning_position(self, position):
        xor = 0
        for c in position:
            xor = c ^ xor
        if xor == 0:
            return False
        else:
            return True

    def legal_moves_on_position(self, position):
        legal_moves = []
        mask = self.get_action_mask(self.position_to_state(position))
        for m, a in zip(mask, self.action_space):
            if m == 1:
                legal_moves.append(a)
        return deepcopy(legal_moves)

    def evaluate_position(self, position):
        winning_move = []
        if self.is_winning_position(position):
            legal_moves = self.legal_moves_on_position(position)
            for move in legal_moves:
                next_position = self.move_on_position(position, move)
                if not self.is_winning_position(next_position):
                    policy = np.zeros((self.action_size,), dtype=np.float64)
                    policy[self.action_space.index(move)] = 1.0
                    winning_move.append(policy)
            return winning_move, 1
        else:
            return [], -1


if __name__ == '__main__':
    nim = NimUnitary(initial_pos=[2, 1, 1])
    state_space, policy_space, value_space, mask_space = nim.get_states_policies_values_masks()

    print('')

