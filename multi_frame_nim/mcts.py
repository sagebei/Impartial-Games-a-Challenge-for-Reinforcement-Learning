import math
import numpy as np
import copy

# The default value for the alpha and epsilon is copied from self-play section of the paper,
# mastering the game of Go without human knowledge
def pucb_score(parent, child, c1=1.25, c2=19652):
    
    pb_c = c1 + math.log((parent.visit_count + c2 + 1) / c2)
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
    prior_score = pb_c * child.prior

    if child.visit_count > 0:
        value_score = - child.value()
    else:
        value_score = 0
    return value_score + prior_score


class Node:
    def __init__(self, prior, to_play):
        self.visit_count = 0  # the times that the node has been visited
        self.to_play = to_play  # the player who takes move at this node
        self.prior = prior  # the prior probability obtained from the policy network (RNN)
        self.value_sum = 0  # the value of the node
        self.children = {}  # save all the child node that can be reached out from this node
        self.state = None  # the board corresponding to this node
        
    def expanded(self):
        return len(self.children) > 0  # the node contains child node if it has been expanded
    
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count  # the value of the node is the mean evaluation over these simulations.

    # play
    def select_action(self, temperature=1):
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = [action for action in self.children.keys()]
        if temperature == 0:  # choose the action with the highest visit count
            action_probs = np.zeros_like(visit_counts)
            action_probs[np.argmax(visit_counts)] = 1.0
            action = np.random.choice(actions, p=action_probs)

        elif temperature == float('inf'):  # choose the action randomly
            action = np.random.choice(actions)

        else:  # choose the action in light of the visit counts adjusted by the temperature
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
            action = np.random.choice(actions, p=visit_count_distribution)
        return action

    # select
    def select_child(self):  # opt for the child node that has the highest UCB score among its siblings.
        best_score = -np.inf
        best_action = -1
        best_child = None

        for i, (action, child) in enumerate(self.children.items()):
            score = pucb_score(self, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child
    
    def expand(self, state, to_play, actions, action_probs):  # expand the node
        self.to_play = to_play
        self.state = state
        for action, prob in zip(actions, action_probs):
            # Only breed the children corresponding to the legal moves
            if prob != 0.0:
                self.children[action] = Node(prior=prob, to_play=self.to_play * -1)

    def add_dirichlet_noise(self, alpha, epsilon):
        actions = list(self.children.keys())
        noise = np.random.dirichlet([alpha] * len(actions))
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - epsilon) + n * epsilon


class MCTS:
    def __init__(self, game, model, args):
        self.game = copy.deepcopy(game)
        self.model = model
        self.args = args
    
    def run(self, state, to_play, is_train=True):
        root = Node(0, to_play)  # initialize the root node

        self.game.reset_board(state.copy())  # initialize the game using the current board
        self.game.player = to_play  # set the first play of the game to the play who is currently at play
        
        action_probs, _ = self.model.predict(state)  # obtain the probability distribution over all the actions.

        action_masks = self.game.action_masks()  # obtain the action mask
        action_probs = action_probs * action_masks  # mask out the illegal actions
        action_probs /= np.sum(action_probs)   # re-normalize the probabilities for remaining moves

        root.expand(state, to_play, self.game.all_legal_actions, action_probs)  # expand the root node
        if is_train:
            root.add_dirichlet_noise(self.args['alpha'], epsilon=self.args['epsilon'])

        for _ in range(self.args['num_simulations']):
            node = root
            search_path = [node]  # a list storing all the child nodes encountered during the simulation

            while node.expanded():  # expand the node if it has not been expanded
                action, node = node.select_child()
                search_path.append(node)  # add the child node to the traversing list.

            parent = search_path[-2]  # the parent node of the leaf node
            self.game.reset_board(parent.state.copy())
            self.game.player = parent.to_play

            # take the action that leads to the child awaiting to be expanded.
            next_state, reward, done = self.game.step(action)
            # If the action results in winning the game at the next_state, this means that the state at which the action
            # is taken is a losing position. Thus, the value of the state is the opposite of the obtained reward.

            # If the next_state is not the terminal state, expand the node that represents the next_state
            if not done:
                action_probs_pred, value = self.model.predict(next_state)
                action_masks = self.game.action_masks()
                action_probs = action_probs_pred * action_masks  # mask out illegal moves
                action_probs /= np.sum(action_probs)
                node.expand(next_state, self.game.to_play(), self.game.all_legal_actions, action_probs)
            else:
                value = -reward

            # update all the nodes at the end of the simulation
            self.backpropagate(search_path, value, self.game.to_play())
        
        return root

    # assign values to the nodes and increment the counter value in the traversing list
    def backpropagate(self, search_path, value, to_play):
        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1



if __name__ == "__main__":
    from game import Nim
    from model import Nim_Model
    args = {
        'n_piles': 3,  # 6, 7
        'num_simulations': 10000,  # 70, 100
        'batch_size': 128,
        'numEps': 104,
        'numIters': 2000,
        'epochs': 1,
        'lr': 0.02,
        'milestones': [200, 600],
        'scheduler_gamma': 0.1,
        'weight_decay': 1e-4,
        'hidden_size': 16,
        'num_lstm_layers': 1,  # 2
        'num_head_layers': 1,
        'branching_factor': 1,
        'exploration_moves': 3,
        'num_samples': 10000,
        'alpha': 0.35,
        'epsilon': 0.25,
        'calculate_elo': False
    }

    game = Nim(n_piles=args['n_piles'])
    model = Nim_Model(action_size=game.action_size,
                      hidden_size=args['hidden_size'],
                      num_lstm_layers=args['num_lstm_layers'],
                      num_head_layers=args['num_head_layers'])
    mcts = MCTS(game, model, args)
    root = mcts.run(game.reset(), game.to_play())
    print([(game.unpack_action(action), child.visit_count) for action, child in root.children.items()])
    
