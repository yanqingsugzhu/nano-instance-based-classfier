import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from LeNet5_bkbone import LeNet5
import numpy as np

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

try:
    model = torch.load('opt_lenet5.pth')
except ModuleNotFoundError:
    print("We cannot find the optimal LeNet5 model for feature extraction!\n"
          "Please train LeNet5 before feature extraction, and do not remove the opt_lenet5.pth \n"
          "I mean run LeNet5_trainer.py before run this python codes")

img_features_training_set = [[], [], [], [], [], [], [], [], [], []]
for data, target in tqdm(train_loader):
    batch_of_img_features = model.feature_extraction(data)
    for id, img_feature in zip(target, batch_of_img_features):
       img_features_training_set[id].append(img_feature)

img_features_testing_set = [[], [], [], [], [], [], [], [], [], []]
for data, target in tqdm(test_loader):
    batch_of_img_features = model.feature_extraction(data)
    for id, img_feature in zip(target, batch_of_img_features):
       img_features_testing_set[id].append(img_feature)


for i in range(10):
    np.save(np.array(img_features_training_set[i]), f"img_features_training_set_{i}.npy")
    np.save(np.array(img_features_testing_set[i]), f"img_features_testing_set_{i}.npy")




