import argparse
import inspect
import os
import pathlib
import shutil
import tempfile
from typing import Dict, List, Optional

import lightgbm as lgb
import mlflow
import pandas as pd
import pytorch_lightning as pl
from joblib import Parallel, delayed
from lightgbm.callback import CallbackEnv, _format_eval_result
from loguru import logger
from matplotlib import pyplot as plt
from mlflow.lightgbm import _patch_metric_names
from tqdm import tqdm

from stock_statistical_arbitrage.strategy import StatisticalArbitrageStrategy
from stock_statistical_arbitrage.utils import init_obj_dict, read_from_yaml


def ts_mean(df, window=10):
    """Computes the rolling mean for the given window size.

    Args:
        df (pd.DataFrame): tickers in columns, dates in rows.
        window      (int): size of rolling window.

    Returns:
        pd.DataFrame: the mean over the last 'window' days.
    """
    return df.rolling(window).mean()


def ts_corr(x, y, window=10):
    """
    Wrapper function to estimate rolling correlations.
    :param x, y: pandas DataFrames.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return x.rolling(window).corr(y)


class BaseFactorMaker:
    """A Basic Factor Set for benchmarking"""

    def __init__(self) -> None:

        # get alpha methods
        methods = inspect.getmembers(self, predicate=inspect.ismethod)
        self.alpha_index, self.alpha_methods = [], []
        for index, method in methods:
            if index.startswith("alpha"):
                self.alpha_index.append(index)
                self.alpha_methods.append(method)

    def __call__(self, dataset: pd.DataFrame) -> pd.DataFrame:

        logger.info("preparing data for feature engineering...")
        logger.info("parallel computation...")
        raise NotImplementedError

    def alpha016(self):
        """avg turnover of 21 / avg turnover of 255"""
        alpha = ts_mean(self.turnover, 21) / ts_mean(self.turnover, 255)
        return alpha.stack().reindex(self.index).to_numpy()

    def alpha027(self):
        """correlation  of 10 days (volume, close) series"""
        alpha = ts_corr(self.volume, self.close, 10)
        return alpha.stack().reindex(self.index).to_numpy()


class LightgbmProfilerCallback:
    """Callback to do signal analysis after each iteration

    FIXME: Reference -> lgb.callback and mlflow.lightgbm private api, some hacking maybe not stable in the future
    """

    def __init__(
        self,
        prefix: str,
        period: int,
        train_index: pd.MultiIndex,
        val_index: pd.MultiIndex,
        strategy: StatisticalArbitrageStrategy,
    ) -> None:
        self.prefix = prefix
        self.period = period
        self.train_index = train_index
        self.val_index = val_index
        self.strategy = strategy

    def __call__(self, env: CallbackEnv):
        if self.period > 0 and (env.iteration + 1) % self.period == 0:
            # NOTE: predict method somehow transfer the model to CPU, won't work
            # Dig deep into lightgbm source code to find this _Booster__inner_predict function

            # log eval results to terminal
            result = "\t".join(
                [_format_eval_result(x, True) for x in env.evaluation_result_list]
            )
            logger.info(f"[{env.iteration + 1}]\t{result}")

            # collect regression metrics
            reg_metrics = {}
            for data_name, eval_name, value, _ in env.evaluation_result_list:
                key = self.prefix + "-" + data_name + "-" + eval_name
                reg_metrics[key] = value
            reg_metrics = _patch_metric_names(reg_metrics)
            mlflow.log_metrics(
                metrics=reg_metrics, step=(env.iteration + 1) // self.period
            )

            # collect portfolio metrics
            train_signal = pd.Series(
                data=env.model._Booster__inner_predict(0), index=self.train_index
            )
            val_signal = pd.Series(
                data=env.model._Booster__inner_predict(1), index=self.val_index
            )
            mlflow.log_metrics(
                metrics={
                    f"{self.prefix}-train-{k}": v
                    for k, v in self.strategy.quick_analysis(
                        signal=train_signal
                    ).items()
                },
                step=(env.iteration + 1) // self.period,
            )
            mlflow.log_metrics(
                metrics={
                    f"{self.prefix}-val-{k}": v
                    for k, v in self.strategy.quick_analysis(signal=val_signal).items()
                },
                step=(env.iteration + 1) // self.period,
            )


class BaseLightgbmStrategy(StatisticalArbitrageStrategy):
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
            dataset.target_type: str
            cv.cv_strategy: ["simple_split", "time_series_split", "custom_split"]
            cv.test_size: float passed when simple_split
            cv.n_train_day/n_val_day: int passed when time_series_split
            model: a lightgbm LGBMRegressor object
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
        Train a lightgbm model, steps:
        1. set up experiment tracker
        2. build features & target
        3. cross validation split
        4. train all splits:
            4.1 init train/val data
            4.2 init model
            4.3 set up logger & callbacks
            4.4 set up the trainer
            4.5 fit the model
            4.6 inference & save results
        5. backtest overall performance
        """
        # 1. set up experiment tracker
        pl.seed_everything(self.train_config["experiment"].pop("seed"))
        _ = self.set_tracker(**self.train_config["experiment"])
        params = pd.json_normalize(self.params).loc[0].to_dict()
        params.pop(
            "train_config.cv.custom_calendar", None
        )  # FIXME: max log length is 500
        mlflow.log_params(params=params)

        # 2. dataset processing: feature/target engineering & stock selection
        features = pd.read_parquet(self.train_config["dataset"]["features_path"])
        target = self.make_target(self.train_config["dataset"]["target_type"])
        target = target[~target.isna()]

        index = self.stock_selection().intersection(target.index)
        features, target = features.reindex(index), target.reindex(index)

        # 3. cross validation split
        trade_calendar = target.index.unique("trade_date")
        n_splits, cv_calendar = self.get_cv_calendar(
            trade_calendar=trade_calendar, **self.train_config["cv"]
        )

        # 4. train all splits
        signals: List[pd.DataFrame] = []
        for split_id in range(0, n_splits):

            # 4.1 select data for current split
            prefix = f"S{split_id+1}"
            logger.info(f"start split {split_id+1}, {n_splits} splits in total...")

            train_calendar, val_calendar = cv_calendar[split_id]
            train_features = features.query("trade_date in @train_calendar")
            train_target = target.reindex(train_features.index)

            val_features = features.query("trade_date in @val_calendar")
            val_target = target.reindex(val_features.index)

            # 4.2 init callbacks
            callbacks = [
                LightgbmProfilerCallback(
                    prefix=prefix,
                    period=self.train_config["callbacks"]["period"],
                    train_index=train_target.index,
                    val_index=val_target.index,
                    strategy=self,
                ),
            ]
            if self.train_config["callbacks"]["patience"] > 0:
                callbacks.append(
                    lgb.callback.early_stopping(
                        first_metric_only=True,
                        stopping_rounds=self.train_config["callbacks"]["patience"],
                    )
                )

            # 4.3 init and fit the model
            model: lgb.LGBMRegressor = init_obj_dict(self.train_config["model"])
            model.fit(
                train_features,
                train_target,
                eval_set=[(train_features, train_target), (val_features, val_target)],
                eval_names=["train", "val"],
                eval_metric=["mse"],
                callbacks=callbacks,
            )

            # 4.4 inference and save results (optional)
            signal = pd.DataFrame(
                data={"signal": model.predict(val_features)}, index=val_target.index
            )
            signals.append(signal)
            if self.train_config["save_things"]:
                tmp_dir: str = tempfile.mkdtemp()
                # save the model
                model.booster_.save_model(
                    os.path.join(tmp_dir, f"{prefix}-best_model.txt")
                )

                # save importance plot
                lgb.plot_importance(
                    model, importance_type="gain", figsize=(20, 16), dpi=800
                )
                plt.savefig(os.path.join(tmp_dir, f"{prefix}-feature_importance.png"))

                # save predictions
                signal.to_parquet(os.path.join(tmp_dir, f"{prefix}-signal.parquet"))

                mlflow.log_artifacts(tmp_dir)
                shutil.rmtree(tmp_dir)

        # 5. evaluate overall performance
        logger.info("evaluating overall performance...")
        signals: pd.DataFrame = pd.concat(signals, axis=0).sort_index()
        mlflow.set_tags(self.quick_analysis(signal=signals))

    @staticmethod
    def make_base_factor_features(src: str, dst: str, start_date: int, end_date: int):
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config_file", help="path to the config file", required=True, type=str
    )
    args = parser.parse_args()
    config = read_from_yaml(args.config_file)
    strategy = BaseLightgbmStrategy(**config)
    strategy.setup()
    strategy.train()
