import torch
from models.model import CAMSATransformer
from utils.prepare_dataset import prepare_imdb
from utils.masks import create_stride_mask
import config
from sklearn.metrics import classification_report, confusion_matrix

device = torch.device("cpu")

# Load dataset
train_loader, test_loader, vocab = prepare_imdb("data/aclImdb")

# Load model
model = CAMSATransformer(len(vocab))
model.load_state_dict(torch.load("camsa_model.pt", map_location=device))
model.to(device)
model.eval()


def generate_stride_masks(seq_len):
    masks = []
    for stride in config.STRIDES:
        mask = create_stride_mask(seq_len, stride).to(device)
        masks.append(mask)
    return masks


stride_masks = generate_stride_masks(config.MAX_SEQ_LEN)

all_preds = []
all_labels = []

with torch.no_grad():
    for x, y in test_loader:

        x = x.to(device)
        y = y.to(device)

        outputs = model(x, stride_masks)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())


print("\nClassification Report:")
print(classification_report(all_labels, all_preds))

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))