import torch
import numpy as np
import random
from game import Nim
from model import Nim_Model
from trainer import Trainer
import ray

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

num_workers = 20 
args = {
    'init_board': [2 for _ in range(10)],
    'include_history': False,
    'num_simulations': 20, 
    'batch_size': 64,
    'numEps': num_workers * 10,
    'numIters': 3000,
    'epochs': 2,
    'lr': 0.02,
    'milestones': [200, 600],
    'scheduler_gamma': 0.1,
    'weight_decay': 1e-4,
    'hidden_size': 16,
    'num_lstm_layers': 1, 
    'num_head_layers': 1,
    'alpha': 0.35,
    'epsilon': 0.25,
}
args['model_dir'] = f"./models/{len(args['init_board'])}_{args['include_history']}_{args['alpha']}"

if __name__ == '__main__':
    set_seed(30)

    ray.init(ignore_reinit_error=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_id = "_".join(str(p) if not isinstance(p, list) else "m".join(str(i) for i in p) for p in args.values())

    game = Nim(init_board=args['init_board'], 
               include_history=args['include_history'])
    model = Nim_Model(action_size=game.action_size,
                      hidden_size=args['hidden_size'],
                      num_lstm_layers=args['num_lstm_layers'],
                      num_head_layers=args['num_head_layers'])

    trainer = Trainer(game, model, args, device, num_workers=num_workers)
    trainer.learn()

    # save the model after the final update
    model.save_checkpoint(args['model_dir'], filename="final")

    ray.shutdown()
