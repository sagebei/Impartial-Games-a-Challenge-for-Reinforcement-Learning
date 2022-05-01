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
    num_workers = 7  # multiprocessing.cpu_count() - 1

    args = {
        'piles': 5,  # 6, 7
        'num_simulations': 50,  # 70, 100
        'batch_size': 128,
        'numIters': 200,
        'numEps': 105,
        'lr': 0.01,
        'scheduler_step_size': 200,
        'scheduler_gamma': 0.1,
        'weight_decay': 1e-4,
        'epochs': 3,
        'hidden_size': 128,
        'num_layers': 1,  # 2
        'branching_factor': 1,
        'exploration_moves': 3,
        'num_samples': 10000,
        'alpha': 0.35,
        'epsilon': 0.25,
        'c_puct': 3,
        'calculate_elo': False
    }

    writer = SummaryWriter(f'logs/{args["piles"]}_{args["alpha"]}_{args["c_puct"]}_{args["num_layers"]}_{args["numIters"]}_{args["num_simulations"]}_{args["numEps"]}')

    game = NimEnv(num_piles=args['piles'])
    model = Nim_Model(action_size=game.action_size,
                      hidden_size=args['hidden_size'],
                      num_layers=args['num_layers'])

    trainer = Trainer(game, model, args, writer, device, num_workers=num_workers)
    trainer.learn()

    # save the model after the final update
    model.save_checkpoint('./models', filename=f'{args["piles"]}_final')

    writer.close()
    ray.shutdown()
