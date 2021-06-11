import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from imagedataset import ImageDataset


device = 'cuda' if torch.cuda.is_available() else 'cpu'


MODEL_NAME = 'model1623440121.pkl'
BATCH_SIZE = 64

print('Loading model...')

model = torch.load('models/' + MODEL_NAME).to(device).eval()
criterion = nn.CrossEntropyLoss().to(device)

df = pd.read_csv('data_resized/test.csv')
dataloader = DataLoader(ImageDataset(df), batch_size=BATCH_SIZE, shuffle=True)

print('Evaluating...')

losses = []
accs = []
for batch in dataloader:
    inputs = batch['image']
    labels = batch['labels']
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    losses.append(loss.item())
    outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1)
    accs.append(np.mean(labels.cpu().detach().numpy() == outputs))

print(f'Overall accuracy:  {"%.3f" % np.mean(accs)}')
print(f'Overall loss:      {"%.3f" % np.mean(losses)}')

num_cls = 7

count_per_cls = np.zeros(num_cls).astype(int)
fr_num = np.zeros(num_cls).astype(int)
fa_num = np.zeros(num_cls).astype(int)
tp_num = np.zeros(num_cls).astype(int)
for batch in dataloader:
    inputs = batch['image']
    labels = batch['labels'].cpu().detach().numpy()
    outputs_prob = model(inputs).cpu().detach().numpy()
    outputs = np.argmax(outputs_prob, axis=1)
    for i in range(len(labels)):
        count_per_cls[labels[i]] += 1
        if labels[i] != outputs[i]:
            fr_num[labels[i]] += 1
            fa_num[outputs[i]] += 1
        else:
            tp_num[labels[i]] += 1

for c in range(num_cls):
    print(f'Class: {c},  share: {"%.3f" % (count_per_cls[c] / count_per_cls.sum())},  '
          f'FR rate: {"%.3f" % (fr_num[c] / count_per_cls[c])},  '
          f'FA rate: {"%.3f" % (fa_num[c] / count_per_cls[c])},  '
          f'F1 score: {"%.3f" % (tp_num[c] / (tp_num[c] + (fr_num[c] + fa_num[c]) / 2))}')
