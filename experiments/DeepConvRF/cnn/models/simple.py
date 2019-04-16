import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN1layer(nn.Module):

    def __init__(self, num_filters, num_classes):
        super(SimpleCNN1layer, self).__init__()
        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=10, stride=2)
        self.fc1 = nn.Linear(144*num_filters, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
        x = F.log_softmax(self.fc1(x), dim=1)
        return(x)


class SimpleCNN2Layers(nn.Module):

    def __init__(self, num_filters, num_classes):
        super(SimpleCNN2Layers, self).__init__()
        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=10, stride=2)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=7, stride=1)
        self.fc1 = nn.Linear(36*num_filters, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
        x = F.log_softmax(self.fc1(x), dim=1)
        return(x)
