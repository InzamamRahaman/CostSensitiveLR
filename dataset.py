import torch.utils.data as data_utils


class PairedDataset(data_utils.Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y
        self.n = len(y)

    def __len__(self):
        return self.n

    def __getitem__(self, item):
        return self.X[item], self.y[item]
