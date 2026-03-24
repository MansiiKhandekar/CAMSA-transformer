import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import matplotlib.pyplot as plt
from models.model import CAMSATransformer
from utils.masks import create_stride_mask
from utils.prepare_dataset import prepare_imdb
import config

device = torch.device("cpu")

# Load vocab
_, _, vocab = prepare_imdb("data/aclImdb")

# Load model
model = CAMSATransformer(len(vocab))
if config.USE_CAMSA:
 model.load_state_dict(torch.load("camsa_model.pt", map_location=device))
else:
   model.load_state_dict(torch.load("standard_model.pt", map_location=device))  
model.to(device)
model.eval()

def generate_stride_masks(seq_len):
    masks = []
    for stride in config.STRIDES:
        mask = create_stride_mask(seq_len, stride).to(device)
        masks.append(mask)
    return masks

stride_masks = generate_stride_masks(config.MAX_SEQ_LEN)

# Sample input
text = "this movie was surprisingly good and very enjoyable"

tokens = text.split()
indices = [vocab.get(t, 0) for t in tokens]

# pad
if len(indices) < config.MAX_SEQ_LEN:
    indices += [0] * (config.MAX_SEQ_LEN - len(indices))
else:
    indices = indices[:config.MAX_SEQ_LEN]

x = torch.tensor(indices).unsqueeze(0).to(device)

# forward pass
with torch.no_grad():
    _ = model(x, stride_masks)

# get attention
attn = model.layers[-1].attn.last_attention[0].cpu().numpy()

seq_len = len(text.split())
attn = attn[:seq_len, :seq_len]
seq_len = len(text.split())
attn = attn[:seq_len, :seq_len]

# plot
plt.imshow(attn, cmap="viridis")
plt.colorbar()
if config.USE_CAMSA:
 plt.title("CAMSA Attention Heatmap")
else:
    plt.title("Standard Attention Heatmap")
plt.xlabel("Tokens")
plt.ylabel("Tokens")
plt.show()