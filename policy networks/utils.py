import random
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Dataset(Dataset):
    def __init__(self, n_samples, n_piles):
        self.n_samples = n_samples
        self.n_piles = n_piles
        self.X = list()
        self.Y = list()
        self.build_dataset()
        print(Counter(self.Y))


    def __len__(self):
        return len(self.X)

    def generate_data(self):
        x = np.random.randint(2, size=2 * self.n_piles)
        x = np.insert(x, [2 + 3 * i - i for i in range(self.n_piles - 1)], -1)

        f_vals = x[[0 + 3 * i for i in range(self.n_piles)]]
        s_vals = x[[1 + 3 * i for i in range(self.n_piles)]]

        f = sum(f_vals) % 2
        s = sum(s_vals) % 2

        if f == 0 and s == 0:
            return None, None
        elif f == 1 and s == 0:
            return x, 0
        elif f == 0 and s == 1:
            return x, 1
        elif f == 1 and s == 1:
            return x, 2

    def build_dataset(self):
        for _ in range(self.n_samples):
            x, y = self.generate_data()
            if x is not None:
                x = np.expand_dims(x, axis=-1).astype(np.float32)
                self.X.append(x)
                self.Y.append(y)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


def batch_accuracy(y_pred_batch, y_batch):
    y_pred_batch = torch.argmax(y_pred_batch, dim=1, keepdim=False)
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
            y_pred = model(X_batch)
            batch_acc = batch_accuracy(y_pred, y_batch)
            accuracy.append(batch_acc)
    model.train()
    if len(accuracy) == 0:
        return 0
    return sum(accuracy) / len(accuracy)


if __name__ == '__main__':
    data = Dataset(n_samples=1000, n_piles=5)
    print(data.X, data.Y)
