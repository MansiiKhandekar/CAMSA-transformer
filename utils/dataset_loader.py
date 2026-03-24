# utils/dataset_loader.py

import os
import torch
from torch.utils.data import Dataset
import config as config
from utils.tokenizer import encode


class IMDBDataset(Dataset):

    def __init__(self, texts, labels, vocab):

        self.texts = texts
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def pad_sequence(self, seq):

        if len(seq) >= config.MAX_SEQ_LEN:
            return seq[:config.MAX_SEQ_LEN]

        padding = [0] * (config.MAX_SEQ_LEN - len(seq))
        return seq + padding

    def __getitem__(self, idx):

        text = self.texts[idx]
        label = self.labels[idx]

        ids = encode(text, self.vocab)

        ids = self.pad_sequence(ids)

        return torch.tensor(ids), torch.tensor(label)


def load_imdb_split(folder):

    texts = []
    labels = []

    for label in ["pos", "neg"]:

        path = os.path.join(folder, label)

        for file in os.listdir(path):

            with open(os.path.join(path, file), encoding="utf8") as f:

                texts.append(f.read())

            labels.append(1 if label == "pos" else 0)

    return texts, labels