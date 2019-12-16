import torch
import torch.utils.data

import model_simple as model

from train import MAX_TRAJ_LENGTH


def main(noise_size, hidden_size, model_path, dataset_size, batch_size):

    device = ["cpu", "cuda"][torch.cuda.is_available()]

    G = model.Generator(noise_size, hidden_size, MAX_TRAJ_LENGTH).to(device)

    z = torch.randn(dataset_size, noise_size)


