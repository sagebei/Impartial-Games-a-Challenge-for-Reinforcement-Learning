import torch
import numpy as np
import random
from NimEnvironments import NimEnv
from model import Nim_Model
from trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
import multiprocessing
import ray


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    ray.init(ignore_reinit_error=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(30)
    num_workers = 8  # multiprocessing.cpu_count() - 1

    initial_pos = [2]
    initial_pos.extend([1 for _ in range(1, 20)])
    print(initial_pos)

    args = {
        'initial_position': initial_pos,  # 6, 7
        'num_simulations': 50,  # 70, 100
        'batch_size': 128,
        'numEps': 104,
        'numIters': 2000,
        'epochs': 1,
        'lr': 0.02,
        'milestones': [200, 600],
        'scheduler_gamma': 0.1,
        'weight_decay': 1e-4,
        'hidden_size': 128,
        'num_lstm_layers': 1,  # 2
        'num_head_layers': 1,
        'branching_factor': 1,
        'exploration_moves': 3,
        'num_samples': 10000,
        'alpha': 0.35,
        'epsilon': 0.25,
        'calculate_elo': False
    }
    
    train_id = "_".join(str(p) if not isinstance(p, list) else "n".join(str(i) for i in p) for p in args.values())
    writer = SummaryWriter("logs/"+train_id)

    game = NimEnv(initial_pos=args['initial_position'])
    game.reset()

    model = Nim_Model(action_size=game.action_size,
                      hidden_size=args['hidden_size'],
                      num_lstm_layers=args['num_lstm_layers'],
                      num_head_layers=args['num_head_layers'])

    trainer = Trainer(game, model, args, writer, device, num_workers=num_workers)
    trainer.learn()

    # save the model after the final update
    model.save_checkpoint('./models', filename=train_id)

    writer.close()
    ray.shutdown()
