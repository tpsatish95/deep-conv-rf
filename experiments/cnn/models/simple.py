import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN1layer(nn.Module):

    def __init__(self, num_filters, num_classes, img_shape=(32, 32, 3)):
        super(SimpleCNN1layer, self).__init__()
        img_dim, _, num_channels = img_shape

        self.conv1 = nn.Conv2d(num_channels, num_filters, kernel_size=10, stride=2)
        self.fc1 = nn.Linear((self.compute_fc_dim(img_dim)**2)*num_filters, num_classes)

    def compute_fc_dim(self, img_dim):
        c1 = int((img_dim - self.conv1.kernel_size[0]) / self.conv1.stride[0]) + 1
        return c1

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
        x = F.log_softmax(self.fc1(x), dim=1)
        return(x)


class SimpleCNN2Layers(nn.Module):

    def __init__(self, num_filters, num_classes, img_shape=(32, 32, 3)):
        super(SimpleCNN2Layers, self).__init__()
        img_dim, _, num_channels = img_shape

        self.conv1 = nn.Conv2d(num_channels, num_filters, kernel_size=10, stride=2)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=7, stride=1)
        self.fc1 = nn.Linear((self.compute_fc_dim(img_dim)**2)*num_filters, num_classes)

    def compute_fc_dim(self, img_dim):
        c1 = int((img_dim - self.conv1.kernel_size[0]) / self.conv1.stride[0]) + 1
        c2 = int((c1 - self.conv2.kernel_size[0]) / self.conv2.stride[0]) + 1
        return c2

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
        x = F.log_softmax(self.fc1(x), dim=1)
        return(x)
