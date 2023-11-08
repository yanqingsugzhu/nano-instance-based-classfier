import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from predictor import Predictor

BATCH_SIZE = 25
EPOCH = 1000
LEARNING_RATE = 1e-3
MONITOR_PATH = "" # Training log path
DATASET_PATH = "" # Dataset path
LABEL_PATH = "" # Label path


def dna_to_one_hot(dna_sequence):
    DNA_MAP = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    return np.array([DNA_MAP[x] for x in dna_sequence], dtype=np.float32)


class DNASequenceDataset(Dataset):
    def __init__(self, dataset_path, label_path):
        with open(dataset_path, 'r') as f:
            self.sequences = [dna_to_one_hot(line.strip()) for line in f.readlines()]

        with open(label_path, 'r') as f:
            self.labels = [float(line.strip()) for line in f.readlines()]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# dataset initialization
dataset = DNASequenceDataset(dataset_path=DATASET_PATH, label_path=LABEL_PATH)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# model initialization
model = Predictor()

# loss func and optimizer initialization
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# train the model
model.train()
for epoch in range(EPOCH):
    for i, (inputs, labels) in enumerate(tqdm(dataloader)):
        outputs = model(inputs)
        loss = criterion(outputs, labels.view(-1, 1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 10 == 0:
            with open(f"{MONITOR_PATH}/training_log.txt", "a+") as log_writer:
                log_writer.write(f'Epoch [{epoch + 1}/{100}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}\n')
            print(f'Epoch [{epoch + 1}/{100}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')

# ±£´æÄ£ÐÍ
torch.save(model.state_dict(), f'{MONITOR_PATH}/predictor_model.pth')
