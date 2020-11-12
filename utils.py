from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import random
import numpy as np
import torch.nn as nn
import cv2


def encoder_decoder(classes):
    decoder = {}
    for i in range(len(classes)):
        decoder[classes[i]] = i

    encoder = {}
    for i in range(len(classes)):
        encoder[i] = classes[i]

    return encoder, decoder


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


def class_plot(data, encoder, inv_normalize=None, n_figures=12):
    print('Printing random data from dataset')
    n_row = int(n_figures/3)
    fig, axes = plt.subplots(figsize=(14, 10), nrows=n_row, ncols=3)
    for ax in axes.flatten():
        a = random.randint(0, len(data))
        (image, label) = data[a]
        label = int(label)
        l = encoder[label]
        if(inv_normalize != None):
            image = inv_normalize(image)

        image = image.numpy().transpose(1, 2, 0)
        im = ax.imshow(image)
        ax.set_title(l)
        ax.axis('off')
    plt.show()
