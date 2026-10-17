import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transformers
from torch.utils.data import DataLoader
transform = transformers.Compose([transformers.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

class CNN_with_FC(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ) 
        self.classifer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.feature(x)
        x = self.classifer(x)
        return x

model = CNN_with_FC()

model.to('cuda')
credits = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 3

print("start traning")
for epoch in range(epochs):
    runing_loss = 0.0
    for i, (image, label) in enumerate(train_loader):
        optimizer.zero_grad()
        image = image.to('cuda')
        label = label.to('cuda')
        outputs = model(image)
        loss = credits(outputs, label)
        loss.backward()
        optimizer.step()
        runing_loss += loss.item()

        if(i + 1) % 300 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {runing_loss/300:.4f}')
            running_loss = 0.0

print("end traing")

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model.eval()

correct = 0
total = 0


with torch.no_grad():
    for images, labels in test_loader:
        images = images.to('cuda')
        labels = labels.to('cuda')
        
        outputs = model(images)
        
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total