import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from pytorch_lightning.loggers import WandbLogger

from src import utils
from src.features.build_features import MyLittleDataModule

from src.models.model_architecture import ResNet18_4ch



class MyLittleLightningModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # self.model = smp.UnetPlusPlus(
        #     "resnet34", encoder_weights=None, in_channels=4, classes=1
        # )
        # self.loss = smp.losses.DiceLoss("binary")

        self.model = ResNet18_4ch(pretrained=True, requires_grad=True)

        self.loss = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        rgb, depth, keypoints = batch
        combined = torch.cat((rgb, depth), dim=1)
        logits = self.model(combined)
        loss = self.loss(logits, keypoints.squeeze())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        rgb, depth, keypoints = batch
        combined = torch.cat((rgb, depth), dim=1)
        logits = self.model(combined)
        loss = self.loss(logits, keypoints.squeeze())
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


def main():
    utils.set_seed(42)
    utils.login_wandb()

    dm = MyLittleDataModule()
    model = MyLittleLightningModule()
    trainer = pl.Trainer(
        gpus=0, logger=WandbLogger(project="GALIROOT"), log_every_n_steps=5
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
