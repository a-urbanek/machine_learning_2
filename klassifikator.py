import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Definiere eine Transformation, die die Daten in Tensoren umwandelt und normalisiert
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

# Lade die Trainings- und Testdaten
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)

# Definiere das Modell
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

# Definiere den Optimierer und die Verlustfunktion
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# Trainiere das Modell
for epoch in range(10):  # Anzahl der Epochen
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # Lade die Eingabe- und Ziel-Daten
        inputs, labels = data

        # Setze die Parameter-Gradienten auf Null
        optimizer.zero_grad()

        # Vorhersagen des Modells berechnen
        outputs = net(inputs)

        # Berechne den Verlust
        loss = criterion(outputs, labels)

        # Berechne die Gradienten
        loss.backward()

        # Optimiere das Modell
        optimizer.step()

        # Verlust des Mini-Batchs speichern
        running_loss += loss.item()
        if i % 100 == 99:    # alle 100 Mini-Batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')

# Teste das Modell auf den Testdaten
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
