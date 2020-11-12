from clr import *
from torchsummary import summary
from model import classifier
from utils import class_plot
from utils import encoder_decoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import av
import torchvision
from torchvision import models, transforms
from torch.autograd import Variable
import sys
from tqdm.autonotebook import tqdm
import torch.optim as optim
import os
from torch.utils.data import DataLoader, Dataset


im_size = 128
batch_size = 8


mean, std = np.array([0.5755, 0.5755, 0.5754]), np.array(
    [0.0993, 0.0993, 0.0993])

# image transformation for train_data
train_transforms = transforms.Compose([
    transforms.Resize((im_size, im_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])

# inverse normalization for image plot

inv_normalize = transforms.Normalize(
    mean=-1*np.divide(mean, std),
    std=1/std
)


# load the train data using Imagefolder of torchvision
train_data = torchvision.datasets.ImageFolder(
    root='dataset', transform=train_transforms)
classes = train_data.classes
# print(classes)

# import encoder and decoder for classes from utils

encoder, decoder = encoder_decoder(classes)
# print(encoder,decoder)

# plot the image using class_plot function in utils.py
# class_plot(train_data, encoder, inv_normalize)

# loading the data using pytorch dataloader
train_loader = DataLoader(
    train_data, batch_size=batch_size, num_workers=8, shuffle=True)

# save it in a dictionary
dataloaders = {'train': train_loader}


model = classifier(n_class=4, device='cuda', pretrained=True)

# summary(model, input_size=(3, 128, 128))


device = 'cuda'
criterion = nn.CrossEntropyLoss().to('cuda')
optimizer = torch.optim.SGD(
    model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
num_epochs = 20
onecyc = OneCycle(len(train_loader)*num_epochs, 1e-3)


for epoch in range(num_epochs):
    print('')
    print(f"--- Epoch {epoch+1} ---")
    phase1 = dataloaders.keys()
    for phase in phase1:
        print('')
        print(f"--- Phase {phase} ---")
        epoch_metrics = {"loss": [], "acc": []}
        for batch_i, (X, y) in enumerate(dataloaders[phase]):
            image = Variable(X.to(device), requires_grad=True)
            labels = Variable(y.to(device), requires_grad=False)
            optimizer.zero_grad()
            predictions = model(image)
            loss = criterion(predictions, labels)
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
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f (%f), Acc: %.2f%% (%.2f%%)]"
                % (
                    epoch + 1,
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
        torch.save(model.state_dict(), '/content/{}.h5'.format(epoch))
        if(phase == 'train'):
            acc_all.append(np.mean(epoch_metrics["acc"]))
            loss_all.append(np.mean(epoch_metrics["loss"]))
