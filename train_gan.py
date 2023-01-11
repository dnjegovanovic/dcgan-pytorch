from models.GAN import GANModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from dataset import dataset
import torch
import os

if __name__ == "__main__":
    PATH_DATASETS = r"./dataset/data"
    BATCH_SIZE = 256 if torch.cuda.is_available() else 64
    NUM_WORKERS = int(os.cpu_count() / 2)

    dm = dataset.Data(PATH_DATASETS, BATCH_SIZE, NUM_WORKERS)
    gan_model = GANModel(*dm.dims)
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        max_epochs=5,
        callbacks=[TQDMProgressBar(refresh_rate=20)],
    )
    trainer.fit(gan_model, dm)
