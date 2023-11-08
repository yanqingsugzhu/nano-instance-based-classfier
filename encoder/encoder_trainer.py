import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from predictor.predictor import Predictor
import numpy as np
from loss import get_seq_loss, get_encoding_loss

BATCH_SIZE = 25
EPOCH = 1000
LEARNING_RATE = 1e-3
MONITOR_PATH = "" # Training log path
IMG_FEATURE_PATH = "" # image feature path


def dna_to_one_hot(dna_sequence):
    DNA_MAP = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    return np.array([DNA_MAP[x] for x in dna_sequence], dtype=np.float32)


class FeatureVecDataset(Dataset):
    def __init__(self, img_fv_path, cls):
        self.fvs = np.load(f"{img_fv_path}/img_feature_{cls}.npy")

    def __len__(self):
        return len(self.fvs)

    def __getitem__(self, idx):
        return fvs[idx]


# dataset initialization
dataloaders = []
for i in range(10):
    dataloaders.append(DataLoader(FeatureVecDataset(img_fv_path=DATASET_PATH, cls=i), batch_size=BATCH_SIZE, shuffle=True))

# model initialization
model = Encoder()

# optimizer initialization
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# train the model
model.train()
for epoch in range(EPOCH):
    for i, (inputs, labels) in enumerate(tqdm(dataloader)):
        outputs = model(inputs)
        loss = get_seq_loss(outputs) + get_encoding_loss(outputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 10 == 0:
            with open(f"{MONITOR_PATH}/training_log.txt", "a+") as log_writer:
                log_writer.write(f'Epoch [{epoch + 1}/{100}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}\n')
            print(f'Epoch [{epoch + 1}/{100}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')

# ±£´æÄ£ÐÍ
torch.save(model.state_dict(), f'{MONITOR_PATH}/encoder_model.pth')
