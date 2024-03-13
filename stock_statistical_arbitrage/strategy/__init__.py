import datetime
import pathlib
from typing import Dict, List, Optional

import cvxpy as cp
import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from tqdm import tqdm

from stock_statistical_arbitrage.utils import annualized_return, annualized_sharp_ratio


class StatisticalArbitrageStrategy:
    """
    Statistical arbitrage strategy for Chinese stock market, a.k.a `pure alpha strategy`
    This class is a template which defines common workflows.
    """

    BASE_COLUMNS = [
        # index
        "stock_code",
        "trade_date",
        # state
        "list_date",
        "st",
        "paused",
        # market
        "open",
        "close",
        "high",
        "low",
        "vwap",
        "volume",
        "up_limit",
        "down_limit",
        "adjusted_open",
        "adjusted_close",
        "adjusted_vwap",
        # barra
        "BETA",
        "MOMENTUM",
        "SIZE",
        "EARNYILD",
        "RESVOL",
        "GROWTH",
        "BTOP",
        "LEVERAGE",
        "LIQUIDTY",
        "SIZENL",
        "Agriculture",
        "Automobiles",
        "Banks",
        "BasicChemicals",
        "BuildMater",
        "CateringTourism",
        "Coal",
        "Computers",
        "Conglomerates",
        "Construction",
        "Defense",
        "ElectronicCompon",
        "FoodBeverages",
        "HealthCare",
        "HomeAppliances",
        "LightIndustry",
        "Machinery",
        "Media",
        "NonbankFinan",
        "NonferrousMetals",
        "Petroleum",
        "PowerEquip",
        "PowerUtilities",
        "RealEstate",
        "RetailTrade",
        "Steel",
        "Telecoms",
        "TextileGarment",
        "Transportation",
        "ConsumerServices",
        "DiverseFinan",
        "Electronics",
        "PowerEquipNewEnergy",
        "COUNTRY",
    ]

    def __init__(
        self,
        holding_period: int,
        exec_price: str,
        base_dataset_path: str,
        index_weight_paths: Optional[Dict[str, str]] = None,
        backtest_config: Optional[dict] = None,
    ) -> None:
        """
        Parameters
        ----------
        holding_period: days to hold the position for a period
        exec_price: execution price, open/close/vwap
        base_dataset_path: path to base dataset, contains data for train, backtest and analysis
        index_weight_paths: paths to index weights, for portfolio construction
        backtest_config: parameters for backtest
        """
        self.holding_period = holding_period
        self.exec_price = exec_price
        self.base_dataset_path = base_dataset_path
        self.index_weight_paths = index_weight_paths
        self.backtest_config = backtest_config

    def setup(self):
        """
        prepare data for training, backtest and analysis

        Set Attributes
        ----------
        self.base_dataset:
        self.index_weights:
        """
        logger.info("It may take minutes to set up.")
        raise NotImplementedError

    def stock_selection(self) -> pd.MultiIndex:
        """
        select stocks according to following rules:
        ...

        Returns
        ----------
        stock_valid: pd.MultiIndex of valid index (stock_code, trade_date) tuple
        """
        raise NotImplementedError

    def make_target(self, target_type: str) -> pd.Series:
        """
        make a regression target with following choice:
        ...

        Returns
        ----------
        target: target for prediction model
        """
        assert hasattr(self, "base_dataset"), "Run setup() before make_target()."
        raise NotImplementedError

    def quick_analysis(self, signal: pd.Series) -> Dict[str, float]:
        """
        evaluate performance of alpha signal quickly

        Returns
        ---------
        metrics: performance metrics
        """
        raise NotImplementedError

    def simulation_analysis(self, signal: pd.Series) -> Dict[str, float]:
        """
        evaluate performance of alpha signal by simulation

        Returns
        ---------
        metrics: performance metrics
        """
        raise NotImplementedError

    @staticmethod
    def set_tracker(tracking_uri: str, experiment_name: str, run_name: str) -> int:
        """setup mlflow tracker

        Returns
        ----------
        run_id: current run_id of the run
        """
        raise NotImplementedError

    @staticmethod
    def make_base_dataset(src: str, dst: str, start_date: int, end_date: int) -> None:
        """
        make base_dataset for backtest and analysis, it contains information of:
        ...
        """
        raise NotImplementedError

    @staticmethod
    def get_residual(dataset: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    @staticmethod
    def get_cv_calendar(
        trade_calendar: List[pd.Timestamp],
        cv_strategy: str,
        test_size: Optional[float] = None,
        n_train_day: Optional[int] = None,
        n_val_day: Optional[int] = None,
        custom_calendar: Optional[List] = None,
        **kwargs,
    ):
        """get cross validation calendar"""
        raise NotImplementedError

    @staticmethod
    def optimize_step(
        excess_return: np.ndarray,
        index_weight: np.ndarray,
        risk_exposure: np.ndarray,
        index_bias: float = 0.01,
        industry_bias: float = 0.01,
        stype_bias: float = 0.01,
    ) -> np.ndarray:
        """generate one step portfolio weight"""
        raise NotImplementedError
