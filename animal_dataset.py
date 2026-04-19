import os.path

import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose, Resize, ToPILImage


class AnimalDataset(Dataset):

    def __init__(self, root, transform=None, train=True ):
        file_path = os.path.join(root, 'animalstt')
        data_path = os.path.join(file_path, 'train' if train else 'test')

        self.transform = transform
        self.images_list =[]
        self.labels_list = []
        print(os.listdir(data_path))
        for i, category in enumerate(os.listdir(data_path)):
            data_file = os.path.join(data_path, category)
            for item in os.listdir(data_file):
                path = os.path.join(data_file, item)
                self.images_list.append(path)
                self.labels_list.append(i)

    def __len__(self):
        return len(self.labels_list)

    def __getitem__(self, idx):
        image_path = self.images_list[idx]
        image = Image.open(image_path).convert('RGB')
        lable = self.labels_list[idx]
        if self.transform:
            image = self.transform(image)
        return image, lable

if __name__ == '__main__':
    transform = Compose([
        Resize((224, 224)),
        ToTensor()
    ])
    dataset = AnimalDataset(root='data',train=True, transform = transform)
    idx = 1
    image, label = dataset.__getitem__(idx)
    # print(image.shape)
    # print(label)
    # print(dataset)
    plt.imshow(ToPILImage()(image))
    plt.show()
"""
Bài toán: Classification
Input: Image
Output: Class
Label lấy từ folder name
Dataset return: (image, label)
"""