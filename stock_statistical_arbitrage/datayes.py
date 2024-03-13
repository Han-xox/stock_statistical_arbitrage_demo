# -*- coding: utf-8 -*-
import datetime
import pathlib
from typing import List, Optional

import fire
import pandas as pd
import requests
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm


class DataYesDataset:
    """Download & Maintain DataYes database based on file system

    Download:
    NOTE: Two query methods: 1)query latest. 2)query by day.
    NOTE: Data is available after close.
    ----------
    1. Basics:
    * Trade Calendar
    * Stock Info
    * Stock Industry
    * Stock Paused
    * Stock ST

    2. Chinese Equity Market:
    * Stock Price
    * Stock Price Adjusted(后复权)
    * Stock Limit
    * Stock Daily Indicator
    * Stock Evaluation
    * Stock Evaluation Plus
    * Stock Money Flow
    * Stock Money Flow Detail


    3. Chinese Equity Risk Model:
    * Risk Exposure

    4. Chinese Equity Index:
    * Index Price
    * Index Weight (沪深300/中证500/中证1000)

    5. Level-2 Data (Not download by this class)

    Update
    ----------
    1. update basics
    2. update stock market
    3. update risk model

    Make Datasets
    ----------
    1. stock daily dataset
    2. stock intraday dataset
    """

    THE_START = "2010-01-01"
    DATAYES_URL = "https://api.datayes.com/data/v1/"

    def __init__(
        self, token: str, base_url: str, l2_url: Optional[str] = None, n_jobs: int = 32
    ) -> None:
        self.token = token
        self.n_jobs = n_jobs
        self.base_url = pathlib.Path(base_url)
        self.l2_url = pathlib.Path(l2_url)

        self.base_url.mkdir(parents=True, exist_ok=True)
        self.l2_url.mkdir(parents=True, exist_ok=True)

    def _request_data(self, api_url: str):
        # construct the header with token
        headers = {
            "Authorization": "Bearer " + self.token,
            "Accept-Encoding": "gzip, deflate",
        }

        # request from api
        res = requests.request("GET", url=self.DATAYES_URL + api_url, headers=headers)
        code = res.status_code
        result = res.content.decode("utf-8")
        if code == 200 and eval(result)["retCode"] == 1:
            # text -> pd.Dataframe
            return pd.DataFrame(eval(result)["data"])
        else:
            logger.error(f"code:{code}, result:{result}")
            return code, result

    ########## Basics ##########
    def get_calendar(self, exchange: str = "XSHG"):
        """
        Doc:https://mall.datayes.com/datapreview/1293
        NOTE: exchange->XSHG, only return open days
        """
        api_url = f"/api/master/getTradeCal.json?field=&exchangeCD={exchange}&isOpen=1"
        return self._request_data(api_url=api_url)

    def get_stock_info(self):
        """
        Doc:https://mall.datayes.com/datapreview/106
        NOTE: only include Chinese A-share stocks
        """
        api_url = "/api/equity/getEqu.json?field=&equTypeCD=A"
        return self._request_data(api_url=api_url)

    def get_stock_industry(self, trade_date: int):
        """
        Doc:https://mall.datayes.com/datapreview/114
        """

        industry_id = "010317"  # 中信行业
        # intoDate 参数介绍: 输入一个日期，可以获取该日期下的股票所属行业信息，输入格式“YYYYMMDD”。例如输入"20190722"，获取这一天当时生效的行业成分信息。
        api_url = f"/api/equity/getEquIndustry.json?field=&industryVersionCD={industry_id}&intoDate={trade_date}"
        return self._request_data(api_url=api_url)

    def get_stock_paused(self, trade_date: int):
        """
        Doc:https://mall.datayes.com/datapreview/150
        """
        api_url = f"/api/master/getSecHalt.json?field=&beginDate={trade_date}&endDate={trade_date}"
        return self._request_data(api_url=api_url)

    def get_stock_st(self, trade_date: int):
        """
        Doc:https://mall.datayes.com/datapreview/1540
        """
        api_url = f"/api/equity/getSecST.json?field=&beginDate={trade_date}&endDate={trade_date}"
        return self._request_data(api_url=api_url)

    ########## Chinese Equity Market ##########
    def get_stock_price(self, trade_date: int):
        """
        Doc:https://mall.datayes.com/datapreview/80
        NOTE: This API returns B-shares stock, there is a corresponding record from listed to de-listed, isOpen == 1 if volume != 0
        """
        api_url = f"/api/market/getMktEqud.json?field=&tradeDate={trade_date}"
        return self._request_data(api_url=api_url)

    def get_stock_price_adjusted(self, trade_date: int):
        """
        Doc: https://mall.datayes.com/datapreview/1598 (后复权)
        """
        api_url = f"/api/market/getMktEqudAdjAf.json?field=&tradeDate={trade_date}"
        return self._request_data(api_url=api_url)

    def get_stock_limit(self, trade_date: int):
        """
        Doc:https://mall.datayes.com/datapreview/1357

        TODO: This API returns even funds data?

        通联数据对数据的解释
        ----------
        涨跌停表针对暂停上市的股票, 深圳是没有的，上海是有的
        这个主要是由于这两个交易所给提供的文件是这样设计的, 我们没有再做一层处理，两市有差异
        另外, 在2020年的时候交易所发布新规取消暂停上市和恢复上市环节, 加快退市节奏, 所以增量两市场都不会在出现暂停上市这种情况了
        """
        api_url = f"/api/market/getMktLimit.json?field=&tradeDate={trade_date}"
        return self._request_data(api_url=api_url)

    def get_stock_indicator(self, trade_date: int):
        """
        Doc: https://mall.datayes.com/datapreview/2935
        """
        api_url = f"/api/market/getMktEqudInd.json?field=&secID=&tradeDate={trade_date}&chgStatus=&beginDate=&endDate="
        return self._request_data(api_url=api_url)

    def get_stock_eval(self, trade_date: int):
        """
        Doc: https://mall.datayes.com/datapreview/2816
        """
        api_url = f"/api/market/getMktEqudEval.json?field=&tradeDate={trade_date}"
        return self._request_data(api_url=api_url)

    def get_stock_eval_plus(self, trade_date: int):
        """
        Doc: https://mall.datayes.com/datapreview/4219
        """
        api_url = f"/api/market/getMktEqudEvalNew.json?field=&secID=&ticker=&tradeDate={trade_date}&beginDate=&endDate="
        return self._request_data(api_url=api_url)

    def get_stock_money_flow(self, trade_date: int):
        """
        Doc: https://mall.datayes.com/datapreview/1585
        """
        api_url = f"/api/market/getMktEquFlow.json?field=&tradeDate={trade_date}&beginDate=&endDate=&secID=&ticker="
        return self._request_data(api_url=api_url)

    def get_stock_money_flow_detail(self, trade_date: int):
        """
        Doc: https://mall.datayes.com/datapreview/1643
        """
        api_url = f"/api/market/getMktEquFlowOrder.json?field=&secID=&ticker=&beginDate={trade_date}&endDate={trade_date}"
        return self._request_data(api_url=api_url)

    def get_index_price(self, trade_date: int):
        """
        Doc: https://mall.datayes.com/datapreview/90
        """
        api_url = f"/api/market/getMktIdxd.json?field=&exchangeCD=XSHE,XSHG&tradeDate={trade_date}"
        return self._request_data(api_url=api_url)

    def make_index_weight(self, ticker: str):
        """
        Doc: https://mall.datayes.com/datapreview/1640
        NOTE: API returns 10W records at a time, query bu month
        """
        START_YEAR = 2010
        END_YEAR = datetime.datetime.today().year
        index_weight = []
        for year in range(START_YEAR, END_YEAR + 1):
            # Get the first and last day of the year
            first_day = datetime.date(year, 1, 1)
            last_day = datetime.date(year, 12, 31)
            # Format the dates as YYYYMMDD
            first_day_str = first_day.strftime("%Y%m%d")
            last_day_str = last_day.strftime("%Y%m%d")
            api_url = f"/api/idx/getIdxCloseWeight.json?field=&ticker={ticker}&beginDate={first_day_str}&endDate={last_day_str}"
            data = self._request_data(api_url=api_url)
            if isinstance(data, pd.DataFrame):
                index_weight.append(data)

        return pd.concat(index_weight, axis=0)

    def get_risk_exposure(self, trade_date: int):
        """
        Doc: https://mall.datayes.com/datapreview/3831
        NOTE: only A-share stocks are contained
        """
        api_url = (
            f"/api/equity/getRMDy1dExposureZX20.json?field=&tradeDate={trade_date}"
        )
        return self._request_data(api_url=api_url)

    ########## update ##########
    @property
    def trade_calendar(self) -> List[int]:
        """Get trade calendar between start_date and end_date"""
        calendar = self.get_calendar()
        end_date = (datetime.datetime.today() - datetime.timedelta(days=2)).strftime(
            "%Y-%m-%d"
        )
        calendar = calendar.query(
            f"calendarDate >= '{self.THE_START}' and calendarDate <= '{end_date}'"
        )
        calendar = calendar["calendarDate"].str.replace("-", "").astype(int).tolist()
        return calendar

    def _update(self, dtype: str):
        api = getattr(self, f"get_{dtype}")
        path = self.base_url / dtype
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"updating {dtype}...")
        for trade_date in tqdm(self.trade_calendar, desc=f"updating {dtype}"):
            dst: pathlib.Path = path / f"{trade_date}.parquet"
            if dst.exists():
                continue
            else:
                data: pd.DataFrame = api(trade_date=trade_date)
                data.to_parquet(dst)

    def update_basics(self):
        """Update:
        1. trade_calendar
        2. stock info
        """
        logger.info("updating calendar...")
        path = self.base_url / "calendar.parquet"
        self.get_calendar().to_parquet(path)

        logger.info("updating stock info...")
        path = self.base_url / "stock_info.parquet"
        self.get_stock_info().to_parquet(path)

    def update_stock_market(self):
        dtypes = [
            "stock_industry",
            "stock_paused",
            "stock_st",
            "stock_price",
            "stock_price_adjusted",
            "stock_limit",
            "stock_indicator",
            "stock_eval",
            "stock_eval_plus",
            "stock_money_flow",
            "stock_money_flow_detail",
        ]
        for dtype in dtypes:
            self._update(dtype=dtype)

    def update_risk_model(self):
        dtypes = [
            "risk_exposure",
        ]
        for dtype in dtypes:
            self._update(dtype=dtype)

    def update_stock_index(self):
        # 1. update index weight
        path = self.base_url / "index_weight"
        path.mkdir(parents=True, exist_ok=True)
        for ticker in ["000300", "000905", "000852"]:
            logger.info(f"updating index weight of {ticker}...")
            data = self.make_index_weight(ticker=ticker)
            data.to_parquet(path / f"{ticker}.parquet")

        # 2. update index price
        self._update(dtype="index_price")

    def update_all(self):
        self.update_basics()
        self.update_stock_market()
        self.update_risk_model()
        self.update_stock_index()

    ########## make datasets ##########
    def make_stock_daily_dataset(self, dst: str) -> None:

        # create folder for the datasets
        dst = pathlib.Path(dst) / f"datayes_stock_daily_dataset"
        dst.mkdir(exist_ok=True, parents=True)

        # run in parallel
        Parallel(n_jobs=self.n_jobs)(
            delayed(self._make_stock_daily_dataset)(
                base_url=self.base_url, trade_date=trade_date, dst=dst
            )
            for trade_date in tqdm(self.trade_calendar)
        )

    def make_stock_intraday_dataset(
        self, freq: str, start_date: int, end_date: int, dst: str
    ):
        # create folder for the datasets
        dst = pathlib.Path(dst) / f"datayes_stock_intraday_{freq}_dataset"
        dst.mkdir(exist_ok=True, parents=True)

        # check calendar
        calendar = list(
            filter(lambda x: x >= start_date and x <= end_date, self.trade_calendar)
        )

        # run in parallel
        Parallel(n_jobs=self.n_jobs)(
            delayed(DataYesDataset._make_stock_intraday_dataset)(
                base_url=self.base_url,
                l2_url=self.l2_url,
                dst=dst,
                trade_date=trade_date,
                freq=freq,
            )
            for trade_date in tqdm(calendar)
        )

    @staticmethod
    def _make_stock_daily_dataset(
        base_url: pathlib.Path, trade_date: int, dst: pathlib.Path
    ) -> pd.DataFrame:
        """Use data from DataYes, generate a dataset consisting of stock daily information:

        1. basics
        2. industry
        3. paused
        4. special treatment (ST)
        5: price & volume
        6. limit
        7. valuation
        8: money flow
        9. barra risk exposure

        NOTE: Data is time continuous: any stock has a corresponding record for each trade day from IPO to delist
        NOTE: NaN in contained in evaluation data
        """

        # if exits, pass
        file_path = dst / f"{trade_date}.parquet"
        if file_path.exists():
            return

        ## 0. read raw data
        stock_info = pd.read_parquet(base_url / "stock_info.parquet")
        stock_industry = pd.read_parquet(
            base_url / "stock_industry" / f"{trade_date}.parquet"
        )
        stock_paused = pd.read_parquet(
            base_url / "stock_paused" / f"{trade_date}.parquet"
        )
        stock_st = pd.read_parquet(base_url / "stock_st" / f"{trade_date}.parquet")
        stock_limit = pd.read_parquet(
            base_url / "stock_limit" / f"{trade_date}.parquet"
        )
        stock_price = pd.read_parquet(
            base_url / "stock_price" / f"{trade_date}.parquet"
        )
        stock_price_adjusted = pd.read_parquet(
            base_url / "stock_price_adjusted" / f"{trade_date}.parquet"
        )
        stock_eval = pd.read_parquet(base_url / "stock_eval" / f"{trade_date}.parquet")
        stock_eval_plus = pd.read_parquet(
            base_url / "stock_eval_plus" / f"{trade_date}.parquet"
        )
        stock_money_flow = pd.read_parquet(
            base_url / "stock_money_flow" / f"{trade_date}.parquet"
        )
        stock_money_flow_detail = pd.read_parquet(
            base_url / "stock_money_flow_detail" / f"{trade_date}.parquet"
        )
        stock_risk_exposure = pd.read_parquet(
            base_url / "risk_exposure" / f"{trade_date}.parquet"
        )

        ## 1. use latest stock_info to select listed stock on current date
        # NaN for not IPO/delist yet
        THE_END = "2996-12-19"  # My 1000th birthday
        stock_info["listDate"] = stock_info["listDate"].fillna(THE_END)
        stock_info["delistDate"] = stock_info["delistDate"].fillna(THE_END)

        # remove wield stocks
        # FIXME: 一些应该被排除的数据, 是由于股票代码变更
        wield_ticker = [
            "000022",
            "601360",
            "201872",
            "000043",
            "200022",
            "601607",
            "001914",
            "001872",
            "600849",
            "601313",
        ]
        stock_info = stock_info.loc[~stock_info["ticker"].isin(wield_ticker)]

        # select listed stock only
        trade_date = pd.to_datetime(trade_date, format="%Y%m%d")
        stock_info = stock_info.query(
            f"listDate < '{trade_date}' and  '{trade_date}' < delistDate"
        )

        d = {
            "secID": "stock_code",
            "ticker": "ticker",
            "secShortName": "stock_name",
            "listDate": "list_date",
            "exchangeCD": "exchange",
            "ListSector": "sector",
        }
        stock_info = stock_info[d.keys()].rename(columns=d).set_index("stock_code")
        stock_info["list_date"] = pd.to_datetime(
            stock_info["list_date"], format="%Y-%m-%d"
        )

        ## 2. get stock industry
        d = {"secID": "stock_code", "industryName1": "industry"}
        stock_industry = (
            stock_industry[d.keys()].rename(columns=d).set_index("stock_code")
        )

        ## 3. mark st stock with 1 flag
        d = {
            "secID": "stock_code",
        }
        stock_st = stock_st[d.keys()].rename(columns=d).set_index("stock_code")
        stock_st["st"] = 1

        ## 4. mark paused stock with 1 flag
        # NOTE: stock can be paused multiple times in a day, just keep the first record
        d = {
            "secID": "stock_code",
        }
        stock_paused = stock_paused[d.keys()].rename(columns=d).set_index("stock_code")
        stock_paused["paused"] = 1
        stock_paused = stock_paused[~stock_paused.index.duplicated(keep="first")]

        ## 5. price & volume
        d = {
            "secID": "stock_code",
            "preClosePrice": "pre_close",
            "openPrice": "open",
            "highestPrice": "high",
            "lowestPrice": "low",
            "closePrice": "close",
            "vwap": "vwap",
            "turnoverVol": "volume",
            "turnoverValue": "value",
            "dealAmount": "amount",
            "turnoverRate": "turnover",
            "chgPct": "pct_change",
        }
        stock_price = stock_price[d.keys()].rename(columns=d).set_index("stock_code")

        d = {
            "secID": "stock_code",
            "openPrice": "adjusted_open",
            "highestPrice": "adjusted_high",
            "lowestPrice": "adjusted_low",
            "closePrice": "adjusted_close",
            "vwap": "adjusted_vwap",
            "turnoverVol": "adjusted_volume",
        }
        stock_price_adjusted = (
            stock_price_adjusted[d.keys()].rename(columns=d).set_index("stock_code")
        )

        ## 6. price limit
        d = {
            "secID": "stock_code",
            "limitUpPrice": "up_limit",
            "limitDownPrice": "down_limit",
            # "upLimitReachedTimes": "up_limit_reached_times",
            # "downLimitReachedTimes": "down_limit_reached_times",
        }
        stock_limit = stock_limit[d.keys()].rename(columns=d).set_index("stock_code")

        ## 7. evaluation
        d = {
            "secID": "stock_code",
            "marketValue": "market_value",
            "negMarketValue": "neg_market_value",
            "PE": "PE".lower(),
            "PE1": "PE1".lower(),
            "PE2": "PE2".lower(),
            "PB": "PB".lower(),
            "PS": "PS".lower(),
            "PS1": "PS1".lower(),
            "PCF": "PCF".lower(),
            "PCF2": "PCF2".lower(),
            "PCF3": "PCF3".lower(),
            "EV": "EV".lower(),
            "EVEBITDA": "EVEBITDA".lower(),
            "EVSales": "EVSales".lower(),
        }
        stock_eval = stock_eval[d.keys()].rename(columns=d).set_index("stock_code")
        d = {
            "secID": "stock_code",
            "freeMarketValue": "free_market_value",
            "PELYR": "PELYR".lower(),
            "PECLYR": "PECLYR".lower(),
            "PECTTM": "PECTTM".lower(),
            "PBLYR": "PBLYR".lower(),
            "PBLYDGW": "PBLYDGW".lower(),
            "PBDGW": "PBDGW".lower(),
            "PSLYR": "PSLYR".lower(),
            "PCFLYR": "PCFLYR".lower(),
            "PCFOLYR": "PCFOLYR".lower(),
        }
        stock_eval_plus = (
            stock_eval_plus[d.keys()].rename(columns=d).set_index("stock_code")
        )

        # 8. money flow
        d = {
            "secID": "stock_code",
            "moneyInflow": "inflow",
            "moneyOutflow": "outflow",
            "netMoneyInflow": "net_inflow",
            "netInflowRate": "net_inflow_rate",
            "netInflowOpen": "net_inflow_open",
            "netInflowClose": "net_inflow_close",
        }
        stock_money_flow = (
            stock_money_flow[d.keys()].rename(columns=d).set_index("stock_code")
        )
        d = {
            "secID": "stock_code",
            "inflowS": "inflow_s",
            "inflowM": "inflow_m",
            "inflowL": "inflow_l",
            "inflowXl": "inflow_xl",
            "outflowS": "outflow_s",
            "outflowM": "outflow_m",
            "outflowL": "outflow_l",
            "outflowXl": "outflow_xl",
            "netInflowS": "net_inflow_s",
            "netInflowM": "net_inflow_m",
            "netInflowL": "net_inflow_l",
            "netInflowXl": "net_inflow_xl",
            "netRateS": "net_rate_s",
            "netRateM": "net_rate_m",
            "netRateL": "net_rate_l",
            "netRateXL": "net_rate_xl",
            "mainInflow": "main_inflow",
            "mainRate": "main_rate",
        }
        stock_money_flow_detail = (
            stock_money_flow_detail[d.keys()].rename(columns=d).set_index("stock_code")
        )

        # 9. barra risk exposure
        stock_risk_exposure = (
            stock_risk_exposure.rename(columns={"secID": "stock_code"})
            .drop(
                columns=[
                    "exchangeCD",
                    "tradeDate",
                    "secShortName",
                    "ticker",
                    "updateTime",
                ]
            )
            .set_index("stock_code")
        )

        ## 10. data alignment
        # NOTE: different tables have different asset codes
        basic_index = stock_info.index

        # current listed stock for index
        stock_industry = stock_industry.reindex(basic_index)

        # events
        stock_st = stock_st.reindex(basic_index).fillna(0)
        stock_paused = stock_paused.reindex(basic_index).fillna(0)

        # should NOT be any NaN in price tables
        stock_price = stock_price.reindex(basic_index)
        stock_price_adjusted = stock_price_adjusted.reindex(basic_index)

        # NaN in this table caused by stop list (暂停上市) stocks, it is a history problem
        stock_limit = stock_limit.reindex(basic_index)

        # many NaN in valuation tables for unknown reasons
        stock_eval = stock_eval.reindex(basic_index)
        stock_eval_plus = stock_eval_plus.reindex(basic_index)
        stock_money_flow = stock_money_flow.reindex(basic_index)
        stock_money_flow_detail = stock_money_flow_detail.reindex(basic_index)

        stock_risk_exposure = stock_risk_exposure.reindex(basic_index)

        #  merge all information
        dataset = pd.concat(
            [
                stock_info,
                stock_industry,
                stock_st,
                stock_paused,
                stock_price,
                stock_price_adjusted,
                stock_limit,
                stock_eval,
                stock_eval_plus,
                stock_money_flow,
                stock_money_flow_detail,
                stock_risk_exposure,
            ],
            axis=1,
        )
        dataset["trade_date"] = trade_date
        dataset = dataset.reset_index(drop=False)

        dataset.to_parquet(file_path)

    @staticmethod
    def _make_stock_intraday_dataset(
        base_url: pathlib.Path,
        l2_url: pathlib.Path,
        dst: pathlib.Path,
        trade_date: int,
        freq: str,
    ) -> None:
        """Make intraday dataset

        NOTE:only contains kline for now
        """

        # if exits, pass
        file_path = dst / f"{trade_date}.parquet"
        if file_path.exists():
            return

        # select valid tickers
        stock_info = DataYesDataset.read_stock_info(
            path=base_url / "stock_info.parquet", trade_date=trade_date
        )
        valid_ticker = stock_info["ticker"].astype(int)

        # read trades from two exchanges
        trades = pd.concat(
            [
                DataYesDataset.get_trades_from_XSHG(
                    path=l2_url
                    / "L2_data"
                    / "MDL"
                    / str(trade_date)
                    / f"{trade_date}_Transaction.csv.zip",
                    trade_date=trade_date,
                ),
                DataYesDataset.get_trades_from_XSHE(
                    path=l2_url
                    / "L2_data"
                    / "MDL"
                    / str(trade_date)
                    / f"{trade_date}_mdl_6_36_0.csv.zip",
                    trade_date=trade_date,
                ),
            ],
            axis=0,
            ignore_index=True,
        )

        trades = trades.loc[trades["ticker"].isin(valid_ticker)]

        # make kline
        kline: pd.DataFrame = trades.groupby("ticker").apply(
            DataYesDataset._make_kline, trade_date=trade_date, freq=freq
        )

        # write to disk
        kline.to_parquet(file_path)

    @staticmethod
    def read_stock_info(path: str, trade_date: int):
        """Use latest stock_info to select listed stock on current date"""

        stock_info = pd.read_parquet(path)

        # NaN for not IPO/delist yet
        stock_info["listDate"] = stock_info["listDate"].fillna("3088-01-01")
        stock_info["delistDate"] = stock_info["delistDate"].fillna("3088-01-01")

        # remove wield stocks
        # FIXME: 一些应该被排除的数据, 是由于股票代码变更
        wield_ticker = [
            "000022",
            "601360",
            "201872",
            "000043",
            "200022",
            "601607",
            "001914",
            "001872",
            "600849",
            "601313",
        ]
        stock_info = stock_info.query("ticker not in @wield_ticker")

        # select listed stock only
        d = pd.to_datetime(trade_date, format="%Y%m%d")
        stock_info = stock_info.query(f"listDate < '{d}' and  '{d}' < delistDate")

        return stock_info

    @staticmethod
    def get_trades_from_XSHG(path: str, trade_date: int):
        """read & process raw trades data from XSHG

        Returns
        ----------
        df with following columns:
        ticker:
        trade_time:
        trade_price:
        trade_volume:
        direction:
        """

        # read from csv from a zipped file
        df = pd.read_csv(
            path,
            usecols=[
                "SecurityID",
                "TradTime",
                "TradPrice",
                "TradVolume",
                "TradeBSFlag",
            ],
            compression="zip",
            index_col=False,
        )

        # rename columns
        df = df.rename(
            columns={
                "SecurityID": "ticker",  # int
                "TradTime": "trade_time",
                "TradPrice": "trade_price",
                "TradVolume": "trade_volume",
                "TradeBSFlag": "direction",
            }
        )

        # N: not known -> for auction
        df["direction"] = df["direction"].map({"N": 0, "B": 1, "S": 2})

        # select continuous auction time
        time_mask = (df["trade_time"] >= "09:30:00.000") & (
            df["trade_time"] <= "15:00:00.000"
        )
        df = df.loc[time_mask]

        # add trade date
        df["trade_time"] = str(trade_date) + " " + df["trade_time"]
        df["trade_time"] = pd.to_datetime(df["trade_time"])

        df = df.reset_index(drop=True)
        return df

    @staticmethod
    def get_trades_from_XSHE(path: str, trade_date: int):
        """read & process raw trades data from XSHE

        Returns
        ----------
        df with following columns:
        ticker:
        trade_time:
        trade_price:
        trade_volume:
        direction:
        """

        # read from csv from a zipped file
        df = pd.read_csv(
            path,
            usecols=[
                "SecurityID",
                "TransactTime",
                "LastPx",
                "LastQty",
                "ExecType",
                "BidApplSeqNum",
                "OfferApplSeqNum",
            ],
            compression="zip",
            index_col=False,
        )

        # rename columns
        df = df.rename(
            columns={
                "SecurityID": "ticker",  # int
                "LastPx": "trade_price",
                "LastQty": "trade_volume",
                "TransactTime": "trade_time",
                "ExecType": "exec_type",
                "BidApplSeqNum": "bid_order_id",
                "OfferApplSeqNum": "ask_order_id",
            }
        )

        # select continuous auction time & filter cancel order
        df = df.loc[
            (df["trade_time"] >= "09:30:00.000")
            & (df["trade_time"] <= "15:00:00.000")
            & (df["exec_type"] == 70)  # 52 is deal, 72 is cancel
        ]

        # if bid_order_id is larger, than it is bid driven
        df["direction"] = df.eval("bid_order_id > ask_order_id").astype(int) + 1

        # add trade date
        df["trade_time"] = str(trade_date) + " " + df["trade_time"]
        df["trade_time"] = pd.to_datetime(df["trade_time"])

        df = df[["ticker", "trade_time", "trade_price", "trade_volume", "direction"]]

        df = df.reset_index(drop=True)

        return df

    @staticmethod
    def _make_kline(
        df: pd.DataFrame, trade_date: int, freq: str = "15min"
    ) -> pd.DataFrame:

        # set index
        df = df.set_index("trade_time").sort_index()

        # add new columns
        df["trade_num"] = 1
        df["trade_value"] = df["trade_price"] * df["trade_volume"]
        buy_mask = df["direction"] == 1
        sell_mask = df["direction"] == 2
        df["buy_value"] = df["trade_value"].where(buy_mask, 0)
        df["sell_value"] = df["trade_value"].where(sell_mask, 0)
        df["buy_volume"] = df["trade_volume"].where(buy_mask, 0)
        df["sell_volume"] = df["trade_volume"].where(sell_mask, 0)
        df["buy_num"] = df["trade_num"].where(buy_mask, 0)
        df["sell_num"] = df["trade_num"].where(sell_mask, 0)

        # resample frequency
        resampler = df.resample(freq, label="left", closed="left")

        # down sample to bar
        kline = pd.DataFrame()
        kline["open"] = resampler["trade_price"].first()
        kline["close"] = resampler["trade_price"].last()
        kline["high"] = resampler["trade_price"].max()
        kline["low"] = resampler["trade_price"].min()

        kline["trade_value"] = resampler["trade_value"].sum()
        kline["trade_volume"] = resampler["trade_volume"].sum()
        kline["trade_num"] = resampler["trade_num"].sum()

        kline["buy_value"] = resampler["buy_value"].sum()
        kline["buy_volume"] = resampler["buy_volume"].sum()
        kline["buy_num"] = resampler["buy_num"].sum()

        kline["sell_value"] = resampler["sell_value"].sum()
        kline["sell_volume"] = resampler["sell_volume"].sum()
        kline["sell_num"] = resampler["sell_num"].sum()

        # reindex
        trade_date = str(trade_date)
        index = pd.date_range(
            start=trade_date + " " + "9:30:00",
            end=trade_date + " " + "11:30:00",
            freq=freq,
            inclusive="left",
        ).append(
            pd.date_range(
                start=trade_date + " " + "13:00:00",
                end=trade_date + " " + "15:00:00",
                freq=freq,
                inclusive="left",
            )
        )
        index.name = "trade_time"

        kline = kline.reindex(index=index)

        return kline


if __name__ == "__main__":
    # example commands
    # python trader/data/datayes.py --token=XXX --base_url=/data/datayes --l2_url /data/datayes_l2 update_all
    # python trader/data/datayes.py --token=XXX --base_url=/data/datayes --l2_url /data/datayes_l2 make_equity_intraday --freq="15min" --start_date=20160509 --end_date=20231231 --dst=/data/dataset
    fire.Fire(DataYesDataset)
