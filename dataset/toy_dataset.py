from torch.utils.data import Dataset


class ToyDataset(Dataset):
    def __init__(self, sample_list: list, args, split: str):
        self.sample_list = sample_list
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.sample_list)
