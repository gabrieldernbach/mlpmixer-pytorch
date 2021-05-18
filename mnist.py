import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from model import MlpMixer

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_ds = MNIST('data/', train=True, download=True, transform=transform)
test_ds = MNIST('data/', train=False, transform=transform)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=64, shuffle=True)

model = MlpMixer(img_size=28, patch_size=7, n_dim=64, n_classes=10)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    loss_avg = 0.
    for idx, (img, tar) in enumerate(train_dl, start=1):
        out = model(img)
        loss = criterion(input=out, target=tar)
        loss_avg += (loss.item() - loss_avg) / idx

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    correct = 0
    for idx, (img, tar) in enumerate(test_dl):
        with torch.no_grad():
            out = model(img)
            pred = out.argmax(dim=1, keepdim=True)
            correct += pred.eq(tar.view_as(pred)).sum().item()

    s_epoch = f"[{epoch:2d} / {num_epochs}] "
    s_loss = f"{loss_avg=:.12f} "
    s_acc = f"acc: {correct / len(test_dl.dataset)}"
    print(s_epoch + s_loss + s_acc)
