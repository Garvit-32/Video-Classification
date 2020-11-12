from model import resnet50
from torch.autograd import Variable
from clr import *
from torchsummary import summary
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sys


data_path = 'dataset'

classes = os.listdir(data_path)


# id = list()

# for i in os.listdir(data_path):
#     p1 = os.path.join(data_path, i)
#     for j in os.listdir(p1):
#         p2 = os.path.join(p1, j)
#         id.append((i, p2))


# class VideoDataset(Dataset):

#     def __init__(self, frame_list=id, sequence_length=16, transform=None):
#         self.frame_list = frame_list
#         self.sequence_length = sequence_length
#         self.transform = transform

#     def __len__(self):
#         return len(self.frame_list)

#     def __getitem__(self, idx):
#         label, path = self.frame_list[idx]
#         img = cv2.imread(path)
#         for i in range(self.sequence_length):
#             img1 = img[:, 128*i:128*(i+1), :]
#             if(self.transform):
#                 img1 = self.transform(img1)
#             seq_img.append(img1)
#         seq_image = torch.stack(seq_img)
#         seq_image = seq_image.reshape(3, 16, im_size, im_size)
#         return seq_image, decoder[label]


# x = VideoDataset().__getitem__(5)

im_size = 128

# mean = [0.4889, 0.4887, 0.4891]
# std = [0.2074, 0.2074, 0.2074]


mean = np.array([0.5765, 0.5747, 0.5752])

std = np.array([0.1000, 0.1020, 0.1021])

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

train_data = VideoDataset(id, sequence_length=16, transform=train_transforms)
train_loader = DataLoader(train_data, batch_size=8,
                          num_workers=4, shuffle=True)
dataloaders = {'train': train_loader}


def normalization_parameter(dataloader):
    mean = 0.
    std = 0.
    nb_samples = len(dataloader.dataset)
    for data, _ in tqdm(dataloader):
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
    mean /= nb_samples
    std /= nb_samples
    return mean, std


# mean, std = normalization_parameter(train_loader)
# print(mean, std)


class classifier(nn.Module):
    def __init__(self, n_class, pretrained=True):
        super(classifier, self).__init__()
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


# model = classifier(n_class=4).to('cuda')
# # print(model)

model = resnet50(class_num=8).to('cuda')

device = 'cuda'
cls_criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(
    model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
num_epochs = 20
onecyc = OneCycle(len(train_loader) * num_epochs, 1e-3)


os.makedirs('weights_crime', exist_ok=True)
iteration = 0
acc_all = list()
loss_all = list()

for epoch in range(num_epochs):
    print('')
    print(f"--- Epoch {epoch} ---")
    phase1 = dataloaders.keys()
    for phase in phase1:
        print('')
        print(f"--- Phase {phase} ---")
        epoch_metrics = {"loss": [], "acc": []}
        for batch_i, (X, y) in enumerate(dataloaders[phase]):
            #iteration = iteration+1
            image_sequences = Variable(X.to(device), requires_grad=True)
            print(image_sequences.shape)
            labels = Variable(y.to(device), requires_grad=False)
            optimizer.zero_grad()
            # model.lstm.reset_hidden_state()
            predictions = model(image_sequences)
            loss = cls_criterion(predictions, labels)
            acc = 100 * (predictions.detach().argmax(1)
                         == labels).cpu().numpy().mean()
            loss.backward()
            optimizer.step()
            epoch_metrics["loss"].append(loss.item())
            epoch_metrics["acc"].append(acc)
            if(phase == 'train'):
                lr, mom = onecyc.calc()
                update_lr(optimizer, lr)
                update_mom(optimizer, mom)
            batches_done = epoch * len(dataloaders[phase]) + batch_i
            batches_left = num_epochs * len(dataloaders[phase]) - batches_done
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f (%f), Acc: %.2f%% (%.2f%%)]"
                % (
                    epoch,
                    num_epochs,
                    batch_i,
                    len(dataloaders[phase]),
                    loss.item(),
                    np.mean(epoch_metrics["loss"]),
                    acc,
                    np.mean(epoch_metrics["acc"]),
                )
            )

            # Empty cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print('')
        print('{} , acc: {}'.format(phase, np.mean(epoch_metrics["acc"])))
        torch.save(model.state_dict(), 'weights_crime/c3d_{}.h5'.format(epoch))
        if(phase == 'train'):
            acc_all.append(np.mean(epoch_metrics["acc"]))
            loss_all.append(np.mean(epoch_metrics["loss"]))


def error_plot(loss):
    plt.figure(figsize=(10, 5))
    plt.plot(loss)
    plt.title("Training loss plot")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.show()


def acc_plot(acc):
    plt.figure(figsize=(10, 5))
    plt.plot(acc)
    plt.title("Training accuracy plot")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.show()


error_plot(loss_all)
acc_plot(acc_all)
