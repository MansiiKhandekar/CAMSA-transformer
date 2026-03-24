import re
from collections import Counter
import config as config


def tokenize(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    tokens = text.split()
    return tokens


def build_vocab(texts):

    counter = Counter()

    for text in texts:
        tokens = tokenize(text)
        counter.update(tokens)

    most_common = counter.most_common(config.MAX_VOCAB_SIZE)

    vocab = {"<PAD>":0, "<UNK>":1}

    for word, _ in most_common:
        vocab[word] = len(vocab)

    return vocab


def encode(text, vocab):

    tokens = tokenize(text)

    ids = [
        vocab[token] if token in vocab else vocab["<UNK>"]
        for token in tokens
    ]

    return ids