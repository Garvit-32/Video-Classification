from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


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
