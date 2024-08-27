# flake8: noqa: F401
import warnings

import mlflow
import lightning as L
import mlflow.data.dataset
import mlflow.data.filesystem_dataset_source
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
        enable_checkpointing=False,
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
        ],
    )  # сохраняю стату каждый батч

    mlflow.enable_system_metrics_logging()  # CPU/GPU usage, etc
    mlflow.set_tracking_uri("http://localhost:5000")  # логи
    mlflow.set_registry_uri(
        "sqlite:///mlflow-medical-segmentation/logs/mlflow-registry/mlflow.db"
    )  # база данных
    # mlflow.create_experiment(name="first-experiment")
    mlflow.set_experiment("first-experiment")

    with mlflow.start_run(description="Some description", tags={"status": "pending"}):
        # mlflow.set_tag("status", "pending")

        example_image = list(iter(train_dataloader))[0][0][0]
        example_image = example_image.reshape(1, *example_image.shape)
        signature = mlflow.models.infer_signature(
            model_input=example_image.detach().numpy(),
            model_output=model(example_image).detach().numpy(),
        )

        segnet_bce_loss.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        segnet_bce_loss.save_checkpoint(
            r"mlflow-medical-segmentation\\outputs\\\SegNet 1.77M params bce_loss.ckpt"
        )

        mlflow.log_artifact(
            r"mlflow-medical-segmentation\\outputs\\\SegNet 1.77M params bce_loss.ckpt"
        )

        # Log the model
        mlflow.pytorch.log_model(
            model,
            artifact_path="artifact-folder/LightningModel artifact",
            input_example=example_image.detach().numpy(),
            signature=signature,
            registered_model_name="Tracking model final",
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
