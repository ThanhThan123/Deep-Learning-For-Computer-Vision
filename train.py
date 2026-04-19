import os

import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm import tqdm
from models import SimpleCNN
from animal_dataset import AnimalDataset
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 2

    transform = Compose([
        Resize((224,224)),
        ToTensor()
    ])
    model = SimpleCNN().to(device) # 2
    train_dataset = AnimalDataset(
        root='data',
        train=True,
        transform=transform
    )


    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle= True,
        num_workers=4,
    )
    test_dataset = AnimalDataset(
        train = False,
        root = 'data',
        transform = transform
    )
    test_dataloader = DataLoader(
        dataset = test_dataset,
        batch_size=4,
        shuffle = False,
        num_workers=4,
    )
    # Khởi tạo loss
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    # Epoch
    epochs = 10 #2

    # Writer P6
    writer = SummaryWriter("tensorboard")

    # checkpoint # 7
    save_dir = "trained_models"
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_path = None #8
    start_epoch = 0 #8
    best_acc = 0
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]

    for epoch in range(start_epoch,epochs): # 2
        model.train()
        process_bar = tqdm(train_dataloader, colour="green") # 5

        for iter_idx, (images, labels) in enumerate(process_bar):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step = epoch * len(train_dataloader) + iter_idx #6
            writer.add_scalar("Loss/train",loss.item(), global_step) #6
        # Validate/ test
        model.eval()
        total = 0
        correct = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for images, labels in test_dataloader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                predictions = torch.argmax(outputs, dim=1)

                total += labels.size(0)
                correct += ( predictions == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
        cm = confusion_matrix(all_labels, all_predictions)
        # print(cm)
        accuracy = correct / total

        # 7 Check point
        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_acc": best_acc,
        }
        # 7
        if accuracy > best_acc:
            best_acc = accuracy
            checkpoint["best_acc"] = best_acc
            torch.save(checkpoint, os.path.join(save_dir, "best_cnn.pt"))

        process_bar.set_description(f"Epoch {epoch + 1}/{epochs}  - Loss: {loss.item():.4f} - Val Acc: {accuracy: .4f}")



"""

1. Dữ liệu lấy ở đâu?
2. Model nhận input gì?
3. Model output gì?
4. Sai số đo bằng gì?
5. Gradient update thế nào?
6. Mình đánh giá model bằng gì?

# file train
dataset -> dataloader -> model -> loss -> optimizer -> 1 epoch train
# train loop
forward (model) -> loss  -> optimizer + zero_grad -> loss backward -> step
"""
# tensorboard --logdir tensorboard/