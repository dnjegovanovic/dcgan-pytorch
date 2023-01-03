import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        self.img_shape = img_shape
