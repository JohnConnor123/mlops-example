# flake8: noqa: F401
import warnings

import mlflow
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
        "mlflow-medical-segmentation\\logs\\tb_logs"  # the path of your log file.
    )
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", tracking_address])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")


if __name__ == "__main__":
    # --------------------------------
    mlflow.pytorch.autolog()
    # --------------------------------
    # init_tensorboard() # не включаю, т.к. борда нужна как заглушка для параметра logger трейнера
    logger = TensorBoardLogger(
        "mlflow-medical-segmentation\\logs\\tb_logs",
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

    # ---------------------------------------------------------------
    mlflow.set_tracking_uri("http://localhost:5000")  # https (и file) не работают
    # не имеет эффекта, т.к. в настройках mlflow server указан путь для артифактов
    # mlflow.set_registry_uri("sqlite:///mlflow-medical-segmentation/logs/mlflow.db")
    mlflow.enable_system_metrics_logging()
    mlflow.set_experiment("second experiment")
    mlflow.set_experiment_tags({"status": "pending"})

    with mlflow.start_run(description="My some description"):
        segnet_bce_loss.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
    # ---------------------------------------------------------------

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
