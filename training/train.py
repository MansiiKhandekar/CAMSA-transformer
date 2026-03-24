import torch
import torch.nn as nn
from tqdm import tqdm
import config

from models.model import CAMSATransformer
from utils.masks import create_stride_mask
from utils.prepare_dataset import prepare_imdb


def generate_stride_masks(seq_len, device):

    masks = []

    for stride in config.STRIDES:

        mask = create_stride_mask(seq_len, stride)
        mask = mask.to(device)

        masks.append(mask)

    return masks


def evaluate(model, dataloader, stride_masks, device):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for x, y in dataloader:

            x = x.to(device)
            y = y.to(device)

            outputs = model(x, stride_masks)

            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == y).sum().item()
            total += y.size(0)

    accuracy = correct / total

    return accuracy


def train():

    print("Starting training pipeline...")

    device = torch.device(config.DEVICE)

    print("Using device:", device)

    print("Preparing dataset...")

    train_loader, test_loader, vocab = prepare_imdb("data/aclImdb")

    print("Dataset loaded")
    print("Vocabulary size:", len(vocab))

    print("Initializing model...")

    model = CAMSATransformer(len(vocab)).to(device)

    stride_masks = generate_stride_masks(config.MAX_SEQ_LEN, device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE
    )

    print("Model initialized")
    print("Starting training...")

    for epoch in range(config.EPOCHS):

        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")

        model.train()

        total_loss = 0

        progress_bar = tqdm(train_loader)

        for x, y in progress_bar:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            outputs = model(x, stride_masks)

            loss = criterion(outputs, y)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            progress_bar.set_description(
                f"Epoch {epoch+1} Loss {loss.item():.4f}"
            )

        avg_loss = total_loss / len(train_loader)

        accuracy = evaluate(
            model,
            test_loader,
            stride_masks,
            device
        )

        print(
            f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f} | Test Accuracy: {accuracy:.4f}"
        )

    if config.USE_CAMSA:
     torch.save(model.state_dict(), "camsa_model.pt")
    else:
     torch.save(model.state_dict(), "standard_model.pt")

    print("\nTraining complete.")
    print("Model saved ")


if __name__ == "__main__":

    train()