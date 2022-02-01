import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from pytorch_lightning.loggers import WandbLogger
from src import utils
from src.features.build_features import MyLittleDataModule
import matplotlib.pyplot as plt
import wandb


class MyLittleLightningModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            "resnet34", encoder_weights=None, in_channels=4, classes=1
        )
        self.loss = smp.losses.DiceLoss("binary")

    def training_step(self, batch, batch_idx):
        rgb, depth, mask1 = batch
        combined = torch.cat((rgb, depth), dim=1)
        logits = self.model(combined)
        loss = self.loss(logits, mask1)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        rgb, depth, mask1 = batch
        combined = torch.cat((rgb, depth), dim=1)
        logits = self.model(combined)
        loss = self.loss(logits, mask1)
        plt.imsave(
            "src/visualization/image_tests/logits.png",
            logits[0, :, :, :].cpu().squeeze(0).numpy(),
        )
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


def main():
    utils.set_seed(42)
    utils.login_wandb()

    dm = MyLittleDataModule()
    model = MyLittleLightningModule()
    trainer = pl.Trainer(
        gpus=1, logger=WandbLogger(project="GALIROOT"), log_every_n_steps=5
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
