import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

def train_mnist():
    # Lade den MNIST Datensatz                  // Datensatz wird zu Tensor umgewandelt
    train_set = torchvision.datasets.MNIST(
        root='./data', train=True, download=True,
        transform=transforms.ToTensor()
    )

    test_set = torchvision.datasets.MNIST(
        root='./data', train=False, download=True,
        transform=transforms.ToTensor()
    )

    # Erstelle Dataloader-Objekte            // Wird in "berechenbare" Objekte umgewandelt, Also 64 Bilder pro Datensatz
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=64, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=64, shuffle=False
    )

    # Erstelle das neuronale Netzwerk
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(784, 10)           # // Es wird nur eine "lineare" Schicht für das NN verwendet:
                                                    # // 784 Eingänge, 10 Ausgänge

        def forward(self, x):
            x = x.view(-1, 784)
            x = self.fc1(x)
            return x

    net = Net()

    # Definiere Verlustfunktion und Optimierer                  # // Verlustfunktion: Differenz zwischen Vorhersage und
                                                                # // tatsächlichem label --> Ziel: Differenz minimieren
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)      # // Optimizer: (???) Neues Test-Objekt für nächste
                                                                # // Epoche?

    # Trainiere das Modell
    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            # Setze die Gradienten auf Null
            optimizer.zero_grad()

            # Vorwärtspropagation
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # Rückwärtspropagation
            loss.backward()

            # Aktualisiere Gewichte
            optimizer.step()

            # Drucke Statistiken
            running_loss += loss.item()
            if i % 100 == 99:    # Jede 100 Mini-Batchess
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    # Evaluierung des Modells
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Genauigkeit des Netzes auf dem Test-Datensatz: %d %%' % (
        100 * correct / total))

train_mnist()
