# utils/prepare_dataset.py

import os
import torch
from torch.utils.data import DataLoader

from utils.dataset_loader import IMDBDataset, load_imdb_split
from utils.tokenizer import build_vocab
import config as config


def prepare_imdb(data_dir):

    print("Loading training data...")
    train_texts, train_labels = load_imdb_split(
        os.path.join(data_dir, "train")
    )

    print("Loading test data...")
    test_texts, test_labels = load_imdb_split(
        os.path.join(data_dir, "test")
    )

    print("Building vocabulary...")
    vocab = build_vocab(train_texts)

    print("Vocabulary size:", len(vocab))

    train_dataset = IMDBDataset(train_texts, train_labels, vocab)
    test_dataset = IMDBDataset(test_texts, test_labels, vocab)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE
    )

    return train_loader, test_loader, vocab