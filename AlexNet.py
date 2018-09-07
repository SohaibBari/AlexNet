import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Device configuration
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    device=torch.device('cuda:0')
    print("Using GPU")
else:
    device=torch.device('cpu')
    print("Using CPU")

# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 128
learning_rate = 1e-2
dropout=0.5
L2wd=5e-4
sgdmom=0.9

'''[32x32x3] INPUT
[8x8x64] Conv2d: 64, k=11, s=4,p=5
[4x4x64] maxpool2d: 2x2, s=2
[4x4x64] batchnorm2d
[4x4x192] Conv2d: 192, k=5, s=1,p=2
[2x2x192] maxpool2d: 2x2, s=2
[2x2x384] Conv2d: 384, k=3, s=1,p=1
[2x2x256] Conv2d: 256, k=3, s=1,p=1
[2x2x256] Conv2d: 256, k=3, s=1,p=1
[1x1x256] maxpool2d: 2x2, s=2
FC layer 256->10'''

# CIFAR10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='/media/sohaib/DATA/NUST/TUKL/Data/',
                                             train=True,
                                             transform=transforms.ToTensor(),
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10(root='/media/sohaib/DATA/NUST/TUKL/Data/',
                                            train=False,
                                            transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# Convolutional neural network (two convolutional layers)
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=R2),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 4096),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        out = self.layer1(x)
        out=out.view(out.size(0), -1)
        out = self.layer2(out)
        return out

#model = AlexNet(num_classes).to(device)

model = torch.load('model.pt')

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=sgdmom, weight_decay=L2wd)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the training images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model, 'model.pt')