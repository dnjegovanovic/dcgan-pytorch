import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils


class Data:
    def __init__(self, data_path):
        self.data_path = data_path
