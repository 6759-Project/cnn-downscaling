""" CNN architectures similar to Bano Medina et. al.  See README for reference """

from typing import Tuple

import pytorch_lightning as pl
from torch.nn import \
    Conv2d, Flatten, Linear, Sequential, Unflatten, \
    MSELoss

from torch.optim import AdamW

OUTPUT_SHAPE = (64, 64)


class LitSuperresolutionModelWrapper(pl.LightningModule):
    """ A PyTorch Lightning wrapper over PyTorch models for climate model superresolution """

    def __init__(self, model_keyword: str):
        super().__init__()

        self.model = get_model(model_keyword)
        self.loss = MSELoss()

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        coarse, fine = batch
        fine_pred = self.model(coarse)

        loss = self.loss(fine, fine_pred)
        self.log('Train Loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        coarse, fine = batch
        fine_pred = self.model(coarse)

        loss = self.loss(fine, fine_pred)
        self.log('Validation Loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=.0001)  # TODO(jwhite) parameterize LR
        # TODO (jwhite) add scheduler
        return optimizer




CNN_LM = Sequential(
    Conv2d(1, 6, (3,3), padding="same"),
    Conv2d(6, 3, (3,3), padding="same"),
    Conv2d(3, 2, (3,3), padding="same"),
    Flatten(),
    Linear(648, 20),
    Linear(20, OUTPUT_SHAPE[0]*OUTPUT_SHAPE[1]),
    Unflatten(dim=1, unflattened_size=OUTPUT_SHAPE)
)


def get_model(keyword: str):
    """ Given a keyword, returns the associated model class"""

    models = {
        "LM": CNN_LM
    }

    model = models.get(keyword)

    if not model:
        raise ValueError(f"Model keyword '{keyword}' not recognized")
    else:
        return model
