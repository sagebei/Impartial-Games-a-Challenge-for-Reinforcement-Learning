import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ParityDataset(Dataset):
    def __init__(
        self,
        n_samples,
        n_elems,
        n_nonzero_min=None,
        n_nonzero_max=None,
        exclude_dataset=None,
        unique=False,
        model='rnn',
        noise=True,
    ):
        self.n_samples = n_samples
        self.n_elems = n_elems

        self.n_nonzero_min = 1 if n_nonzero_min is None else n_nonzero_min
        self.n_nonzero_max = (
            n_elems if n_nonzero_max is None else n_nonzero_max
        )

        self.model = model
        self.noise = noise
        self.unique = unique
        self.unique_set = set()
        self.val_set = set() if exclude_dataset is None else exclude_dataset.unique_set

        self.X, self.Y = [], []
        if self.n_samples > 0:
            self.build_dataset()

    def __len__(self):
        return self.n_samples

    def generate_data(self):
        while True:
            x = torch.zeros((self.n_elems,))
            n_non_zero = torch.randint(
                self.n_nonzero_min, self.n_nonzero_max, (1,)
            ).item()

            if self.noise:
                x[:n_non_zero] = torch.randint(0, 2, (n_non_zero,)) * 2 - 1
            else:
                x[:n_non_zero] = 1.0

            x = x[torch.randperm(self.n_elems)]

            y = (x == 1.0).sum() % 2

            item = tuple(x.detach().clone().tolist())
            if self.unique:
                if (item not in self.val_set) and (item not in self.unique_set):
                    return x, y.item(), item
            else:
                if item not in self.val_set:
                    return x, y.item(), item

    def build_dataset(self):
        print('Building dataset ...')
        for _ in range(self.n_samples):
            x, y, item = self.generate_data()
            self.X.append(x)
            self.Y.append(y)
            self.unique_set.add(item)

        self.Y = torch.Tensor(self.Y).float()
        if self.model == 'rnn':
            self.X = torch.stack(self.X).float().unsqueeze(dim=-1)
        elif self.model == 'mlp':
            self.X = torch.stack(self.X).float()
        elif self.model == 'cnn':
            self.X = torch.stack(self.X).to(torch.int64) + 1
            self.X = F.one_hot(self.X, num_classes=3).float()

        perm_index = torch.randperm(self.X.size()[0])
        self.X = self.X[perm_index]
        self.Y = self.Y[perm_index]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


def batch_accuracy(y_pred_batch, y_batch):
    acc = ((y_pred_batch > 0) == y_batch).float().mean().item()
    return acc


def dataloader_accuracy(dataloader, model):
    model.eval()
    accuracy = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(X_batch)[:, 0]
            batch_acc = batch_accuracy(y_pred, y_batch)
            accuracy.append(batch_acc)
    model.train()
    if len(accuracy) == 0:
        return 0
    return sum(accuracy) / len(accuracy)


