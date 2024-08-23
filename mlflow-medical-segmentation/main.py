# flake8: noqa: F401
import warnings

import lightning as L
import losses
import torch
import torchmetrics
from datasets import SegmentationDataset
from lightning.pytorch.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateFinder,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from lightning.pytorch.loggers import TensorBoardLogger
from lightning_model import MyLightningModel
from matplotlib import rcParams
from models.SegNet import SegNet_1768k
from models.Unet import UNet_1811k
from preprocessing import preprocess_data
from tensorboard import program

warnings.filterwarnings(category=SyntaxWarning, action="ignore", module="sanitizer")
rcParams["figure.figsize"] = (15, 4)
torch.set_float32_matmul_precision("high")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def score_model(model, metric, data):
    model.eval()  # testing mode
    scores = 0
    for X_batch, Y_label in data:
        Y_pred = model(X_batch.to(DEVICE)).cpu()
        Y_pred = torch.clamp(Y_pred - 0.5, 0, 1).ceil()  # на всякий случай
        scores += metric(Y_pred, Y_label).item()

    return scores / len(data)


def init_tensorboard():
    tracking_address = (
        "mlflow-medical-segmentation\\tb_logs"  # the path of your log file.
    )
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", tracking_address])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")


if __name__ == "__main__":
    init_tensorboard()
    logger = TensorBoardLogger(
        "mlflow-medical-segmentation\\tb_logs",
        name="SegNet_1.77M bce_loss.pth",
        log_graph=False,
    )

    train_dataloader, val_dataloader, test_dataloader = preprocess_data()

    max_epochs = 250
    model = MyLightningModel(SegNet_1768k(), lr=1.5e-3, loss_fn=losses.bce_loss)

    segnet_bce_loss = L.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        log_every_n_steps=1,
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
            EarlyStopping(
                "lr-Adam",
                mode="min",
                stopping_threshold=1e-5,
                patience=max_epochs,
                verbose=1,
            ),
            ModelCheckpoint(monitor="val_IoU_epoch"),
        ],
    )  # сохраняю стату каждый батч

    segnet_bce_loss.fit(
        model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )

    segnet_bce_loss.save_checkpoint(
        "mlflow-medical-segmentation\\outputs\\\
            SegNet 1.77M_params bce_loss val_iou=0.856 test_iou=0.895\
              lr=1.5e-3 patience=13 gamma=0.2 large_augmentation.ckpt"
    )
    print(
        "val_iou:",
        score_model(
            segnet_bce_loss.model.cuda(),
            torchmetrics.classification.MulticlassJaccardIndex(num_classes=2),
            val_dataloader,
        ),
    )
    print(
        "test_iou",
        score_model(
            segnet_bce_loss.model.cuda(),
            torchmetrics.classification.MulticlassJaccardIndex(num_classes=2),
            test_dataloader,
        ),
    )
