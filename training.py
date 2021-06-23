import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import models
from sklearn.model_selection import train_test_split
from imagedataset import ImageDataset
from augment import augment
import time


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def evaluate(model, criterion, epoch, train_data, val_data, num_batches=None):
    model.eval()
    print(f'\rEpoch: {epoch}/{NUM_EPOCHS},\tevaluating model...', end='')
    loss_tv = []
    acc_tv = []
    for data in [train_data, val_data]:
        losses = []
        accs = []
        i = 0
        for batch in data:
            with torch.no_grad():
                inputs = batch['image']
                labels = batch['labels']
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                losses.append(loss.item())
                label = np.argmax(outputs.cpu().detach().numpy(), axis=1)
                accs.append(np.mean(labels.cpu().detach().numpy() == label))
            i += 1
            if num_batches and i == num_batches:
                break
        loss_tv.append(np.mean(losses))
        acc_tv.append(np.mean(accs))
    print(f'\rEpoch: {epoch}/{NUM_EPOCHS},\t'
          f'train loss: {("%.5f"%loss_tv[0])},  '
          f'val loss: {"%.5f"%loss_tv[1]},  '
          f'train acc: {"%.5f"%acc_tv[0]},  '
          f'val acc: {"%.5f"%acc_tv[1]}', end='')
    model.train()
    return *loss_tv, *acc_tv


model1 = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, padding=1),
    nn.ReLU(),

    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(64),
    nn.MaxPool2d(4),
    nn.Dropout2d(0.25),

    nn.Conv2d(64, 128, kernel_size=5, padding=2),
    nn.ReLU(),

    nn.Conv2d(128, 256, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(256),
    nn.MaxPool2d(2),
    nn.Dropout2d(0.25),

    nn.Flatten(),
    nn.Linear(16384, 1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, 7),
).to(device)

model2 = nn.Sequential(
    models.resnet18(pretrained=True),
    nn.ReLU(),
    nn.Linear(1000, 7),
).to(device)
model2[0].conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

model = model1

print('Number of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
criterion = nn.CrossEntropyLoss().to(device)


VAL_SIZE = 0.2
BATCH_SIZE = 64
NUM_EPOCHS = 100
EVAL_BATCHES = 100
EVAL_STEP = 1
SAVE_MODEL_IF_LOSS_IS_LESS_THAN = 10

df = pd.read_csv('data_resized/train.csv')
train_df, val_df = train_test_split(df, test_size=VAL_SIZE)
train_data = ImageDataset(train_df)
val_data = ImageDataset(val_df)
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

num_batches = len(train_data) // BATCH_SIZE

evaluate(model, criterion, 0, train_dataloader, val_dataloader, EVAL_BATCHES)

for epoch in range(1, NUM_EPOCHS + 1):
    print(f'\nEpoch: {epoch}/{NUM_EPOCHS},\ttraining...', end='')
    losses = []
    i = 0
    best_loss = SAVE_MODEL_IF_LOSS_IS_LESS_THAN

    for batch in train_dataloader:
        i += 1
        print(f'\rEpoch: {epoch}/{NUM_EPOCHS},\tbatch: {i}/{num_batches}...', end='')
        inputs = augment(batch['image'])
        labels = batch['labels']
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    if epoch % EVAL_STEP == 0:
        val_loss = evaluate(model, criterion, epoch, train_dataloader, val_dataloader, EVAL_BATCHES)[0]
        if val_loss < best_loss:
            best_loss = val_loss
            name = f'model{int(time.time())}.pkl'
            torch.save(model, 'models/' + name)
            print(f',  saved as {name}', end='')

