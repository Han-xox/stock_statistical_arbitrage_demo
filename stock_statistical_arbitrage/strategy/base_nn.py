import argparse
import os
import pathlib
import shutil
import tempfile
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from loguru import logger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Sampler
from torchmetrics import MeanAbsolutePercentageError, MeanMetric
from tqdm import tqdm
from trader.utils import CategoryEncoder, init_obj_dict, read_from_yaml

from stock_statistical_arbitrage.strategy import StatisticalArbitrageStrategy


class BaseDataset(Dataset):
    """Dataset deals with daily time series features"""

    def __init__(
        self,
        features: pd.DataFrame,
        cat_features: List[str],
        ts_features: List[str],
        cs_features: List[str],
        feature_map: pd.DataFrame,
        target: Optional[pd.Series] = None,
    ) -> None:
        super().__init__()

        # select start_end & end_index from feature_map
        self.feature_map: np.ndarray = feature_map.loc[
            :, ["start_index", "end_index"]
        ].to_numpy()

        # category, time_series and cross_section features
        self.cat_features = torch.tensor(
            features[cat_features].to_numpy(), dtype=torch.int32
        )
        self.ts_features = torch.tensor(
            features[ts_features].to_numpy(), dtype=torch.float32
        )
        self.cs_features = torch.tensor(
            features[cs_features].to_numpy(), dtype=torch.float32
        )

        # get target
        if target is not None:
            self.target = torch.tensor(target.to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.feature_map)

    def __getitem__(self, index: int):

        start_index, end_index = (
            self.feature_map[index, 0],
            self.feature_map[index, 1],
        )

        category_features = self.cat_features[end_index]

        # left close & right open -> [start_end, end_index+1)
        ts_features = self.ts_features[start_index : end_index + 1]
        cs_features = self.cs_features[start_index : end_index + 1]

        numerical_features = np.concatenate(
            [self.time_series_zscore(ts_features), cs_features], axis=1
        )

        # keep the index for analysis during training
        if hasattr(self, "target"):
            target = self.target[index]
            return (
                index,
                category_features,
                numerical_features,
                target,
            )
        else:
            return (
                index,
                category_features,
                numerical_features,
            )

    @staticmethod
    def time_series_zscore(x: np.array):
        # x shape -> [T, C], mu/sigma -> [C,]
        mu = x.mean(axis=0, keepdims=True)
        sigma = x.std(axis=0, keepdims=True)
        return (x - mu) / (sigma + 1e-5)

    @staticmethod
    def time_series_divide(x: np.array):
        # x shape -> [T, C], x[-1] -> [C,]
        return x / x[-1]


class CrossSectionSampler(Sampler):
    """CrossSectionSampler groups samples of same trade date into a batch"""

    def __init__(self, feature_map: pd.DataFrame, shuffle: bool = True) -> None:
        self.shuffle = shuffle
        self.batch_table = (
            feature_map.reset_index(drop=False)
            .groupby("trade_date")
            .apply(lambda df: df.index.tolist())
            .reset_index(drop=True)
        )

    def __iter__(self):
        index = self.batch_table.index.to_numpy()
        if self.shuffle:
            np.random.shuffle(index)  # this function works inplace
        for batch_id in index:
            yield self.batch_table[batch_id]

    def __len__(self):
        return len(self.batch_table)


class BaseModel(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        embedding: Optional[nn.Module] = None,
        loss: str = "MSE",
        lr: float = 0.0001,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.lr = lr
        self.verbose = verbose
        self.loss = loss
        self.encoder = encoder
        self.predictor = predictor
        self.embedding = embedding

        # save results for performance profile
        self.train_predictions = []
        self.val_predictions = []

        # set in runtime for metrics
        self.strategy: StatisticalArbitrageStrategy = None
        self.train_index: pd.MultiIndex = None
        self.val_index: pd.MultiIndex = None

        self.set_metrics()

    def set_metrics(self):
        """metrics for common regression tasks"""
        [
            setattr(
                self,
                f"{mode}_metrics",
                nn.ModuleDict(
                    {
                        "loss": MeanMetric(),
                        "mape": MeanAbsolutePercentageError(),
                    }
                ),
            )
            for mode in ["train", "val"]
        ]

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, category_features, numerical_features):
        h = self.encoder(numerical_features)[:, -1, :]
        if self.embedding is not None:
            h = torch.concat((self.embedding(category_features), h), dim=1)
        pred = self.predictor(h)
        pred = pred.squeeze(-1)
        return pred

    def forward_step(self, batch):
        (
            index,
            category_features,
            numerical_features,
            target,
        ) = batch

        prediction = self(category_features, numerical_features)

        if self.loss == "MSE":
            loss = F.mse_loss(prediction, target, reduction="mean")
        if self.loss == "MAE":
            loss = F.l1_loss(prediction, target, reduction="mean")

        # metrics
        mode = "train" if self.training else "val"
        metrics = getattr(self, f"{mode}_metrics")
        metrics["loss"](loss)
        metrics["mape"](target, prediction)

        self.log_dict(
            {f"{mode}-{k}": v for k, v in metrics.items()},
            on_epoch=True,
            on_step=False,
        )

        # save the prediction
        indexed_prediction = (index.cpu(), prediction.detach().cpu())
        getattr(self, f"{mode}_predictions").append(indexed_prediction)

        return loss

    def training_step(self, batch, batch_idx):
        return self.forward_step(batch)

    def validation_step(self, batch, batch_idx):
        return self.forward_step(batch)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        if len(batch) == 3:
            (index, category_features, numerical_features) = batch
        else:
            (index, category_features, numerical_features, target) = batch

        prediction = self(category_features, numerical_features)

        # return a index, prediction tuple
        return (index.cpu(), prediction.detach().cpu())

    def on_epoch_end(self) -> None:
        mode = "train" if self.training else "val"
        predictions: List[torch.Tensor] = getattr(self, f"{mode}_predictions")
        index = getattr(self, f"{mode}_index")

        # align nn prediction with stock_code & trade_date
        signal = self.get_signal(predictions=predictions)
        signal.index = index

        metrics: Dict[str, float] = self.strategy.quick_analysis(signal=signal)
        metrics = {f"{mode}-{k}": v for k, v in metrics.items()}
        self.log_dict(metrics)
        setattr(self, f"{mode}_predictions", [])

        if self.verbose:
            metrics = self.trainer.callback_metrics
            s = [f"{mode} epoch {self.trainer.current_epoch}"] + [
                f"{k.upper()} {v}" for k, v in metrics.items() if k.startswith(mode)
            ]
            logger.info(" | ".join(s))

    def on_train_epoch_end(self) -> None:
        self.on_epoch_end()

    def on_validation_epoch_end(self) -> None:
        self.on_epoch_end()

    @staticmethod
    def get_signal(predictions: List[torch.Tensor]) -> pd.Series:
        index = torch.cat([t[0] for t in predictions], dim=0)
        prediction = torch.cat([t[1] for t in predictions], dim=0)
        signal: pd.Series = pd.Series(index=index.numpy(), data=prediction.numpy())
        signal = signal.sort_index()

        return signal


class BaseNeuralNetStrategy(StatisticalArbitrageStrategy):
    """Prediction model = stock daily time series features + neural networks"""

    def __init__(
        self,
        holding_period: int,
        exec_price: str,
        base_dataset_path: str,
        index_weight_paths: Optional[Dict[str, str]] = None,
        train_config: Optional[dict] = None,
    ) -> None:
        """
        train_config: configs for training
            experiment.tracking_uri:
            experiment.experiment_name:
            experiment.run_name:
            experiment.seed
            dataset.features_path:
            dataset.window_size: length of sliding window of daily features
            dataset.cat_features:
            dataset.cs_features:
            dataset.ts_features:
            dataset.target_type: str
            cv.cv_strategy: ["simple_split", "time_series_split", "custom_split"]
            cv.test_size: float passed when simple_split
            cv.n_train_day/n_val_day: int passed when time_series_split
            dataloader.batch_size:
            dataloader.num_workers:
            model: pytorch lightning module object
            trainer: pytorch lightning trainer configs
            callbacks.monitor: str metric to track for checkpointing & early stopping
            callbacks.mode: str ["min", "max"]
            callbacks.patience: int early stopping patience, -1 to turn of early stopping
            save_things: bool
        """
        self.params = locals()
        super().__init__(
            holding_period, exec_price, base_dataset_path, index_weight_paths
        )
        self.train_config = train_config

    def train(self):
        """
        Train a neural network with pytorch lightning, steps:
        1. set up experiment tracker
        2. build features & target
        3. build the window index
        4. cross validation split
        5. train all splits:
            5.1 init dataloaders
            5.2 init model
            5.3 set up logger & callbacks
            5.4 set up the trainer
            5.5 fit the model
            5.6 inference & save results
        6. backtest overall performance
        """

        # 1. set up experiment tracker
        torch.set_float32_matmul_precision("medium")
        pl.seed_everything(self.train_config["experiment"].pop("seed"))
        run_id = self.set_tracker(**self.train_config["experiment"])
        params = pd.json_normalize(self.params).loc[0].to_dict()
        params.pop(
            "train_config.cv.custom_calendar", None
        )  # FIXME: max log length is 500
        mlflow.log_params(params=params)

        # 2. dataset processing: feature/target engineering & sample selection
        features = pd.read_parquet(self.train_config["dataset"]["features_path"])
        target = self.make_target(self.train_config["dataset"]["target_type"])

        # 3. data alignment
        # NOTE: this part is quite complicated, be 100% careful!
        # for features, build a map: a lookup table map (stock_code, trade_date) to (start_index, end_index)
        window_size = self.train_config["dataset"]["window_size"]
        feature_map = pd.DataFrame(
            index=features.index, data={"end_index": range(0, len(features))}
        )
        feature_map["start_index"] = feature_map["end_index"] - window_size + 1
        valid_mask = (
            features.isna()
            .any(axis=1)
            .rolling(window=window_size, min_periods=window_size)
            .sum()
            == 0
        ) & ((features.groupby("stock_code").cumcount() + 1) >= window_size)
        feature_map = feature_map[valid_mask]

        # for target, drop NaN
        target = target[~target.isna()]

        # index alignment & reindex -> data alignment
        index = (
            strategy.stock_selection()
            .intersection(feature_map.index)
            .intersection(target.index)
        )
        feature_map = feature_map.reindex(index)
        target = target.reindex(index)

        # 4. cross validation split
        trade_calendar = sorted(list(index.unique("trade_date")))
        n_splits, cv_calendar = self.get_cv_calendar(
            trade_calendar=trade_calendar, **self.train_config["cv"]
        )

        # 5. train all split
        signals: List[pd.DataFrame] = []
        for split_id in range(0, n_splits):

            # 5.1 init dataloaders
            prefix = f"S{split_id+1}"
            logger.info(f"start split {split_id+1}, {n_splits} splits in total...")

            # get train/val index
            train_calendar, val_calendar = cv_calendar[split_id]
            train_feature_map = feature_map.query("trade_date in @train_calendar")
            train_target = target.reindex(train_feature_map.index)
            val_feature_map = feature_map.query("trade_date in @val_calendar")
            val_target = target.reindex(val_feature_map.index)

            # build dataloaders
            logger.info("building dataset & dataloader...")
            train_dataset = BaseDataset(
                features=features,
                feature_map=train_feature_map,
                cat_features=self.train_config["dataset"]["cat_features"],
                ts_features=self.train_config["dataset"]["ts_features"],
                cs_features=self.train_config["dataset"]["cs_features"],
                target=train_target,
            )
            val_dataset = BaseDataset(
                features=features,
                feature_map=val_feature_map,
                cat_features=self.train_config["dataset"]["cat_features"],
                ts_features=self.train_config["dataset"]["ts_features"],
                cs_features=self.train_config["dataset"]["cs_features"],
                target=val_target,
            )

            train_dataloader = self.get_dataloader(
                dataset=train_dataset,
                feature_map=train_feature_map,
                shuffle=True,
                batch_size=self.train_config["dataloader"]["batch_size"],
                num_workers=self.train_config["dataloader"]["num_workers"],
            )
            val_dataloader = self.get_dataloader(
                dataset=val_dataset,
                feature_map=val_feature_map,
                shuffle=False,
                batch_size=self.train_config["dataloader"]["batch_size"],
                num_workers=self.train_config["dataloader"]["num_workers"],
            )

            # 5.2 init model
            model = init_obj_dict(self.train_config["model"])
            model.strategy = self
            model.train_index = train_target.index
            model.val_index = val_target.index

            # 5.3 set up logger & callbacks
            # make a tmp local storage for checkpoints
            tmp_dir: str = tempfile.mkdtemp()
            mlflow_logger = MLFlowLogger(
                tracking_uri=self.train_config["experiment"]["tracking_uri"],
                experiment_name=self.train_config["experiment"]["experiment_name"],
                run_id=run_id,
                prefix=prefix,
            )
            callbacks = [
                ModelCheckpoint(
                    monitor=self.train_config["callbacks"]["monitor"],
                    mode=self.train_config["callbacks"]["mode"],
                    filename=f"{prefix}-checkpoint",
                    dirpath=tmp_dir,
                ),
            ]
            if self.train_config["callbacks"]["patience"] > 0:
                callbacks.append(
                    EarlyStopping(
                        monitor=self.train_config["callbacks"]["monitor"],
                        mode=self.train_config["callbacks"]["mode"],
                        patience=self.train_config["callbacks"]["patience"],
                    )
                )

            # 5.4 set up trainers
            trainer = pl.Trainer(
                **self.train_config["trainer"],
                logger=mlflow_logger,
                callbacks=callbacks,
                num_sanity_val_steps=0,  # debugging should be done before training
            )

            # 5.5 fit the model
            trainer.fit(
                model=model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
            )

            # 5.6 inference & save results(optional)
            # inference with validation dataset
            inference_dataloader = self.get_dataloader(
                dataset=val_dataset,
                feature_map=val_feature_map,
                shuffle=False,
                batch_size=4098,
                num_workers=8,
            )
            # FIXME: what model is used for inference? best or last
            predictions: List[torch.tensor] = trainer.predict(
                model=model, dataloaders=inference_dataloader
            )
            signal: pd.Series = BaseModel.get_signal(predictions=predictions)
            signal.index = val_target.index
            signal = signal.to_frame("signal")
            # save predictions
            signals.append(signal)
            # write to disk
            if self.train_config["save_things"]:
                signal.to_parquet(os.path.join(tmp_dir, f"{prefix}-signal.parquet"))
                mlflow.log_artifacts(tmp_dir)
                shutil.rmtree(tmp_dir)

        # 5. evaluate overall performance
        logger.info("evaluating overall performance...")
        signals: pd.DataFrame = pd.concat(signals, axis=0).sort_index()
        mlflow.set_tags(self.quick_analysis(signal=signals))

    @staticmethod
    def get_dataloader(
        dataset: Dataset,
        feature_map: pd.DataFrame,
        shuffle: bool,
        batch_size: int,
        num_workers: int,
    ):
        if batch_size == -1:
            sampler = CrossSectionSampler(feature_map=feature_map, shuffle=shuffle)
            dataloader = DataLoader(
                dataset=dataset, batch_sampler=sampler, num_workers=num_workers
            )
        else:
            dataloader = DataLoader(
                dataset=dataset,
                shuffle=shuffle,
                batch_size=batch_size,
                num_workers=num_workers,
            )
        return dataloader

    @staticmethod
    def make_stock_daily_features(
        src: str, dst: str, start_date: int, end_date: int
    ) -> None:
        """Make stock daily features(table format) for neural network training.
        For now, only choose limited features: market price/money flow/evaluation.

        Notes:
        Some data-processing work can NOT be done in current step like time series z-score.
        It is left to be processed in runtime.
        """
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config_file", help="path to the config file", required=True, type=str
    )
    args = parser.parse_args()
    config = read_from_yaml(args.config_file)
    strategy = BaseNeuralNetStrategy(**config)
    strategy.setup()
    strategy.train()
