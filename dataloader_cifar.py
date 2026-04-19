# CIFAR10 built-in dataset cho bài toán classification
from torchvision.datasets import CIFAR10

# Chuyển ảnh -> tensor, scale pixel về [0, 1]
from torchvision.transforms import ToTensor

# DataLoader: chia dữ liệu thành batch để train
from torch.utils.data import DataLoader, Dataset

# Custom Dataset nếu muốn tự định nghĩa cách load dữ liệu
# from cifar10_dataset import MyDataset

import numpy as np

if __name__ == '__main__':
    # Tạo training dataset từ CIFAR10
    # train=True: lấy tập train
    # ToTensor(): ảnh -> tensor
    training_data = CIFAR10(root="data", train=True, transform=ToTensor())

    # Có thể thay bằng custom dataset
    # training_data = MyDataset(root="data", train=True)

    # Debug 1 sample nếu cần
    # image, label = training_data.__getitem__(1234)

    # DataLoader:
    # - batch_size=16: mỗi batch 16 ảnh
    # - shuffle=True: xáo trộn dữ liệu
    # - drop_last=True: bỏ batch cuối nếu thiếu mẫu
    training_dataloader = DataLoader(
        dataset=training_data,
        batch_size=16,
        num_workers=4,
        shuffle=True,
        drop_last=True
    )

    # Duyệt từng batch
    for images, labels in training_dataloader:
        # images.shape thường là [16, 3, 32, 32]
        # = [batch, channel, height, width]
        print(images.shape)

        # labels là nhãn class của từng ảnh trong batch
        print(labels)