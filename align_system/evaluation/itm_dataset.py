from torch.utils.data import Dataset


class ITMDataset(Dataset):
    
    def __init__(self, inputs, labels):
        assert len(inputs) == len(labels), 'Inputs and labels must be the same length'
        self.samples = list(zip(inputs, labels))
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]