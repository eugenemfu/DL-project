import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

class ImageDataset(Dataset):
    def __init__(self, annotations_dataframe, label=None):
        self.img_labels = annotations_dataframe
        if not label is None:
            self.img_labels = self.img_labels[self.img_labels['label'] == label]
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = read_image(self.img_labels.iloc[idx, 0]).type(torch.float32).to(self.device)
        labels = torch.tensor(self.img_labels.iloc[idx, 1]).to(self.device)
        sample = {"image": image, "labels": labels}
        return sample