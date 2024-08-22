# flake8: noqa: F401
from time import time

import lightning as L
import torch
import torchmetrics
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torchvision.transforms import ToTensor


class MyLightningModel(L.LightningModule):
    """Класс обертка для подсчета метрик"""

    def __init__(self, model=None, lr=1e-3):
        super().__init__()
        self.model = model
        self.example_input_array = torch.ones((1, 3, 256, 256))
        self.loss_fn = None
        self.lr = lr
        self.IoU = torchmetrics.classification.MulticlassJaccardIndex(num_classes=2)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_ind):
        self.model.train()
        X, y = batch
        outputs = self.model(X)
        loss = self.loss_fn(outputs, y)

        y_pred = torch.where(outputs < 0, 0, 1)
        self.log_dict(
            {"train_loss": loss.mean(), "train_IoU": self.IoU(y_pred, y)},
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

        return loss.mean()

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        X, y = batch

        with torch.no_grad():
            outputs = self.model(X)
            loss = self.loss_fn(outputs, y)

        y_pred = torch.where(outputs < 0, 0, 1)
        self.log_dict(
            {"val_loss": loss.mean(), "val_IoU": self.IoU(y_pred, y)},
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

        if self.current_epoch % 10 == 0 and batch_idx == 0:
            self.visualise_metrics(X, y, y_pred)

        return loss.mean()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer, factor=0.2, patience=17
                ),  # ExponentialLR(optimizer, gamma=0.93)
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def visualise_metrics(self, X, y, y_pred):
        # Visualize tools
        # clear_output(wait=True)
        X, y, y_pred = X.cpu(), y.cpu(), y_pred.cpu()

        n = 6
        for k in range(n):
            plt.subplot(3, n, k + 1)
            plt.imshow(np.rollaxis(X[k].numpy(), 0, 3), cmap="gray")
            plt.title("X")
            plt.axis("off")

            plt.subplot(3, n, k + n + 1)
            plt.imshow(y_pred[k, 0], cmap="gray")
            plt.title("Y_pred")
            plt.axis("off")

            plt.subplot(3, n, k + 2 * n + 1)
            plt.imshow(y[k, 0], cmap="gray")
            plt.title("Y_true")
            plt.axis("off")

        fig = plt.gcf()
        fig.canvas.draw()
        buffer = fig.canvas.buffer_rgba()
        img_array = np.frombuffer(buffer, dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        image_tensor = ToTensor()(img_array)

        self.logger.experiment.add_image(
            "Predictions at val_loader", image_tensor, self.global_step
        )
