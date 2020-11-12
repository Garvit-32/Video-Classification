import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
# from torchsummary import summary


class classifie(nn.Module):
    def __init__(self, n_class, pretrained=True):
        super(classifie, self).__init__()
        self.cnn_arch = models.resnet50(pretrained=pretrained)
        self.linear1 = nn.Linear(1000, 512)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, n_class)

    def forward(self, input):
        am = self.cnn_arch(input)
        out = self.dropout(self.relu(self.linear1(am)))
        out = self.dropout(self.relu(self.linear2(out)))
        out = self.linear3(out)
        return out


def classifier(n_class, device='cpu', pretrained=True):
    model = classifie(n_class, pretrained)
    model.to(device)
    return model


if __name__ == '__main__':
    model = classifier(n_class=4, device='cuda', pretrained=True)
    # summary(model, input_size=(3, 256, 256))
