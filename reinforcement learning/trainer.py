import numpy as np
from random import shuffle
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from NimEnvironments import NimEnv
from monte_carlo_tree_search import MCTS
from EloRating import Elo
from PlayerPool import PlayerPool
from ExpertPolicyValue import get_states_policies_values_masks
import ray
import copy


@ray.remote
class ParameterServer:
    def __init__(self, model, args):
        self.model = copy.deepcopy(model)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=args['lr'],
                                    weight_decay=args['weight_decay'])
        self.scheduler = StepLR(self.optimizer,
                                step_size=int((args['numIters']*args['epochs'])/3),
                                gamma=0.1)
    
    def apply_graident(self, gradients, use_scheduler=False):
        self.optimizer.zero_grad()
        self.model.set_gradients(gradients)
        self.optimizer.step()
        if use_scheduler:
            self.scheduler.step()
        
    def get_weights(self):
        return self.model.get_weights()
    

@ray.remote
class Simulation:
    def __init__(self, game, model, args, ps):
        self.game = copy.deepcopy(game)
        self.model = copy.deepcopy(model)
        self.args = args
        self.ps = ps

    def execute_episode(self):
        current_weight = ray.get(self.ps.get_weights.remote())
        self.model.set_weights(current_weight)
        mcts = MCTS(self.game, self.model, self.args)
        train_examples = []
        
        state = self.game.reset()
        done = False
        n_moves = 0
        with torch.no_grad():
            while not done:
                root = mcts.run(state, self.game.to_play(), is_train=True)
                action_probs = [0.0 for _ in range(self.game.action_size)]
                for action, child in root.children.items():
                    action_probs[action] = child.visit_count
                action_probs = action_probs / np.sum(action_probs)

                train_examples.append((state, action_probs, self.game.to_play()))

                # Set the temperature for the first a few moves to 1, and the remaining moves to 0.
                if n_moves < self.args['exploration_moves']:
                    temp = 1.0
                else:
                    temp = 0.0

                action = root.select_action(temperature=temp)
                next_state, reward, done = self.game.step(action)
                state = next_state

                n_moves += 1

                if done:
                    # Add the terminal state
                    examples = []
                    for history_state, history_action_probs, history_player in train_examples:
                        examples.append((history_state, history_action_probs,
                                         -reward if history_player == self.game.to_play() else reward))
                    return examples


class Trainer:
    def __init__(self, game, model, args, writer, device, num_workers=4):
        self.game = game
        self.model = model
        self.args = args
        self.writer = writer
        self.device = device
        self.batch_counter = 0  # record the number of batch data used during the training process
        self.epoch_counter = 0  # record the number of training epochs
        self.schduler_counter = 0

        self.ps = ParameterServer.remote(self.model, self.args)
        self.num_workers = num_workers
        self.simulations = [Simulation.remote(self.game, self.model, self.args, self.ps) for _ in range(self.num_workers)]
        
        self.elo = Elo(k=16)
        self.player_pool = PlayerPool(self.elo)
        # get some sampled states and their associated winning move and value.
        self.state_space, self.policy_space, self.value_space, self.masks = get_states_policies_values_masks(game.num_piles,
                                                                                                             num_samples=self.args['num_samples'])

    def learn(self):
        for i in range(1, self.args['numIters'] + 1):
            print(f'{i}/{self.args["numIters"]}')
            train_examples = []
            
            for i in range(self.args['numEps'] // self.num_workers):
                examples = ray.get([sim.execute_episode.remote() for sim in self.simulations])
                for exp in examples:
                    train_examples.extend(exp)
            shuffle(train_examples)
            self.train(train_examples)

    def play(self, first_player_model, second_player_model):
        game = NimEnv(num_piles=self.args['piles'])
        first_player_mcts = MCTS(game, first_player_model, self.args)
        second_player_mcts = MCTS(game, second_player_model, self.args)

        state = game.reset()
        done = False
        while not done:
            # the first play takes the first move
            root = first_player_mcts.run(state, game.to_play(), is_train=False)
            action = root.select_action(temperature=0.0)
            next_state, reward, done = game.step(action)
            # if the first play removes the last matches, it wins the game
            if done:
                return reward
            
            root = second_player_mcts.run(next_state, game.to_play(), is_train=False)
            action = root.select_action(temperature=0.0)
            state, reward, done = game.step(action)
            # if the second play removes the last matches, the first player lost the game
            if done: 
                return -reward

    def save_best_model(self):
        self.player_pool.save_best_player()
    
    def elo_rating_update(self):
        if self.player_pool.counter >= 2:
            latest_player_model = self.player_pool.get_latest_player_model()
            latest_player_id = self.player_pool.counter - 1

            # setup the tournament for the latest player against all its ancestors.
            for opponent_id, opponent_model in enumerate(self.player_pool.pool[:-1]):
                # Agent being trained as first player
                reward = self.play(latest_player_model, opponent_model)
                if reward == 1.0:  # the latest player wins
                    self.player_pool.update_elo_rating(latest_player_id, opponent_id)
                else:  # the opponent wins
                    self.player_pool.update_elo_rating(opponent_id, latest_player_id)
                
                # Agent being trained as second player
                reward = self.play(opponent_model, latest_player_model)
                if reward == 1.0:  # the opponent wins
                    self.player_pool.update_elo_rating(opponent_id, latest_player_id)
                else:  # the latest player wins
                    self.player_pool.update_elo_rating(latest_player_id, opponent_id)

    def train(self, examples):
        for n_epochs in range(self.args['epochs']):
            batch_idx = 0
            while batch_idx < len(examples) // self.args['batch_size']:
                current_weight = ray.get(self.ps.get_weights.remote())
                self.model.set_weights(current_weight)
                self.model.to(self.device)
                self.model.train()

                sample_ids = np.random.randint(len(examples), size=self.args['batch_size'])
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))

                boards = torch.FloatTensor(np.array(boards)).contiguous().to(self.device)
                target_pis = torch.FloatTensor(np.array(pis)).contiguous().to(self.device)
                target_vs = torch.FloatTensor(np.array(vs)).contiguous().to(self.device)

                out_pi, out_v = self.model(boards)
                
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v
                
                self.writer.add_scalar('Pi_Loss', l_pi, self.batch_counter)
                self.writer.add_scalar('V_Loss', l_v, self.batch_counter)

                self.batch_counter += 1
                
                self.model.zero_grad()
                total_loss.backward()
                if n_epochs == self.schduler_counter:
                    self.ps.apply_graident.remote(self.model.get_gradients(), use_scheduler=True)
                    self.schduler_counter += 1
                else: 
                    self.ps.apply_graident.remote(self.model.get_gradients(), use_scheduler=False)
                batch_idx += 1

            # Factor
            for bf in range(1, self.args['branching_factor']+1):
                random_policy_acc, policy_acc, value_acc = self.eval_policy_value_acc(branching_factor=bf)
                self.writer.add_scalars(f'Policy_Branching_{bf}',
                                        {'AlphaZero': policy_acc,
                                         'Random': random_policy_acc}, self.epoch_counter)
                if bf == 1:
                    self.writer.add_scalar(f'Value_Accuracy', value_acc, self.epoch_counter)

            # Elo rating
            self.player_pool.add_player(self.model)
            for _ in range(1):
                self.elo_rating_update()
            last_rating = self.player_pool.get_latest_player_rating()
            self.writer.add_scalar('Elo_Rating', last_rating, self.epoch_counter)
            self.epoch_counter += 1

            if self.epoch_counter % 100 == 0:
                self.model.save_checkpoint('./models', filename=f'{self.args["piles"]}_{self.epoch_counter}')

    def eval_policy_value_acc(self, branching_factor=1, value_threshold=1.0):
        self.model.eval()
        with torch.no_grad():
            p_acc = 0
            random_p_acc = 0
            p_total = 0

            v_acc = 0
            v_total = 0
            for state, policies_target, value_target, mask in zip(self.state_space, self.policy_space, self.value_space, self.masks):
                probs, value = self.model(torch.tensor(state, dtype=torch.float32, device=self.device))
                probs = probs.data.cpu().numpy()
                probs = probs * np.array(mask)
                random_probs = np.random.rand(len(probs)) * np.array(mask)
                # if this state has at least one winning move
                if len(policies_target) > 0:
                    # calculate the accuracy of the policy obtained from the policy network
                    for policy in policies_target:
                        policy = np.array(policy, dtype=np.float32)
                        indicies = probs.argsort()[-branching_factor:].tolist()
                        # if the move corresponding to the highest probability is (one of) the winning move
                        if np.where(policy == 1.0)[0][0] in indicies:
                            p_acc += 1
                            break
                    # calculate the accuracy of the random policy
                    for policy in policies_target:
                        policy = np.array(policy, dtype=np.float32)
                        random_indices = random_probs.argsort()[-branching_factor:].tolist()
                        if np.where(policy == 1.0)[0][0] in random_indices:
                            random_p_acc += 1
                            break

                    p_total += 1
                # calculate the accuracy of the random value
                if abs(value_target - value) < value_threshold:
                    v_acc += 1
                v_total += 1

            return float(random_p_acc/p_total), float(p_acc/p_total), float(v_acc/v_total)

    # calculate the loss for the policy network
    def loss_pi(self, targets, outputs):
        loss = - (targets * torch.log(outputs)).sum(dim=1)
        return loss.mean()

    # calculate the loss for the value network
    def loss_v(self, targets, outputs):
        loss = torch.sum((targets - outputs.squeeze()) ** 2) / targets.size()[0]
        return loss
            
            