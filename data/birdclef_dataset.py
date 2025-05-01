import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio

class BirdCLEFDataset(Dataset):
    """
    Expects:
      root/train.csv
      root/train_audio/<primary_label>/<filename>.ogg
    """
    def __init__(self, root: str, csv_file: str = "train.csv", transforms=None):
        self.root = root
        df = pd.read_csv(os.path.join(root, csv_file))
        # Build relative paths like "greani1/XC132190.ogg"
        self.filepaths = (
            df["primary_label"].astype(str)
            + "/"
            + df["filename"]
        ).tolist()
        # Map each unique label string to an integer
        labels = df["primary_label"].astype("category")
        self.labels = labels.cat.codes.tolist()
        self.idx2species = dict(enumerate(labels.cat.categories))
        self.transforms = transforms

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        rel_path = self.filepaths[idx]
        full_path = os.path.join(self.root, "train_audio", rel_path)
        waveform, sr = torchaudio.load(full_path)
        if self.transforms:
            waveform = self.transforms(waveform)
        label = self.labels[idx]
        return waveform, label
