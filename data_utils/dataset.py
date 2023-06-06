import pandas as pd
from torch.utils.data import Dataset


class InputDataset(Dataset):
    def __init__(self, data_file, sequence_column):
        print("Read data from", data_file)
        df = pd.read_csv(data_file)
        print("Data shape", df.shape)
        self.data = [r[sequence_column] for index, r in df.iterrows()]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index]
